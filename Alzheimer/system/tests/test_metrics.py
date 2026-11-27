"""Unit tests of outcome status, Turnbull estimation, net benefit, the confusion matrix and the decision rule."""
import numpy as np
import pandas as pd
import pytest

from mci_risk.config import BAND_THRESHOLDS, DELTA_AUC, NB_MARGIN
from mci_risk.cv import best_lambda, edge_share, folds_by_rid, pair_label
from mci_risk.metrics import (
    categories,
    cm_key,
    confusion,
    confusion_rates,
    decide,
    nb_key,
    net_benefit,
    status,
    summarize,
    turnbull,
    turnbull_risk,
)
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


def test_confusion_by_hand_with_weights():
    # weights 2,1,1,1,3 (the last patient is censored early: w = 0 and must be ignored)
    y = np.array([1, 1, 0, 0, 0, 1])
    w = np.array([2.0, 1.0, 1.0, 1.0, 3.0, 0.0])
    p = np.array([0.9, 0.1, 0.5, 0.2, 0.05, 0.99])
    c = confusion(y, w, p, 0.3)        # flagged: patients 0 (case) and 2 (non-case); total weight 8
    assert c == pytest.approx(dict(tp=2 / 8, fp=1 / 8, fn=1 / 8, tn=4 / 8))
    r = confusion_rates(**c)
    assert r['sensitivity'] == pytest.approx(2 / 3) and r['specificity'] == pytest.approx(4 / 5)
    assert r['ppv'] == pytest.approx(2 / 3) and r['npv'] == pytest.approx(4 / 5)
    assert r['f1'] == pytest.approx(2 * 2 / (2 * 2 + 1 + 1)) and r['flagged'] == pytest.approx(3 / 8)


def test_confusion_rates_handle_empty_cells():
    r = confusion_rates(tp=0.0, fp=0.0, fn=0.4, tn=0.6)     # nobody flagged
    assert np.isnan(r['ppv']) and r['sensitivity'] == 0 and r['specificity'] == 1


def test_summarize_reports_confusion_at_band_thresholds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(float)
    p = np.clip(0.3 * y + rng.uniform(0, 0.6, 200), 0, 1)
    out = summarize(y, np.ones(200), p, np.full(200, y.mean()), calib=False)
    for t in BAND_THRESHOLDS:
        assert sum(out[cm_key(cell, t)] for cell in ('tp', 'fp', 'fn', 'tn')) == pytest.approx(1)
    assert 0 < out['pr_auc'] <= 1


def test_best_lambda_breaks_ties_by_grid_order():
    assert best_lambda({0.1: -5.0, 1.0: -3.0, 10.0: -3.0}) == 1.0


def test_edge_share():
    grid = [0.1, 1.0, 10.0]
    assert edge_share({'M0': {10.0: 3, 1.0: 1}, 'M1': {1.0: 4}}, grid) == {'M0': 0.75, 'M1': 0.0}
