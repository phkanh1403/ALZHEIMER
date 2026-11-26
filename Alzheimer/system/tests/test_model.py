"""Unit tests of the risk models on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from mci_risk.model import (
    CUTS,
    fit_cox,
    fit_interval,
    interval_negloglik,
    interval_problem,
    predict,
    risk_curve,
    sample_times,
)


@pytest.fixture(scope='module')
def synthetic():
    """Interval-censored data from a known model: hazard doubles per unit of x1, x2 has no effect."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, 2))
    T = rng.exponential(900 / np.exp(np.log(2) * x[:, 0]))
    visits = np.arange(0, 2200, 182.5)
    last = rng.choice(visits[3:], size=n)
    upper = np.array([visits[(visits >= t)][0] if t <= lv else np.inf for t, lv in zip(T, last, strict=True)])
    lower = np.array([visits[visits < t].max() if np.isfinite(u) else lv
                      for t, u, lv in zip(T, upper, last, strict=True)])
    return pd.DataFrame(dict(RID=np.arange(n), x1=x[:, 0], x2=x[:, 1], L=lower, R=upper))


def test_analytic_gradient_matches_finite_differences(synthetic):
    p = interval_problem(synthetic, ['x1', 'x2'])
    theta = np.r_[np.full(len(CUTS) - 1, -6.5), 0.3, -0.1]
    args = (p['X'], p['EL'], p['ED'], p['event'], p['wt'], 1.0, 1.0)
    _, grad = interval_negloglik(theta, *args)
    eps = 1e-6
    numeric = np.array([(interval_negloglik(theta + e, *args)[0] - interval_negloglik(theta - e, *args)[0]) / (2 * eps)
                        for e in np.eye(len(theta)) * eps])
    assert np.abs(grad - numeric).max() / np.abs(numeric).max() < 1e-5


def test_fit_recovers_the_direction_of_effects(synthetic):
    m = fit_interval(synthetic, ['x1', 'x2'], lam=0.1)
    assert m['converged']
    beta_per_unit = np.array(m['beta']) / np.array(m['scale'])
    assert beta_per_unit[0] == pytest.approx(np.log(2), abs=0.15)
    assert abs(beta_per_unit[1]) < 0.15


def test_risk_increases_with_time_and_matches_predict(synthetic):
    m = fit_interval(synthetic, ['x1', 'x2'])
    curve = risk_curve(m, synthetic.head(20), [365, 730, 1095])
    assert (np.diff(curve, axis=1) > 0).all()
    assert np.allclose(curve[:, 1], predict(m, synthetic.head(20), 730))


def test_sampled_onsets_lie_inside_their_interval(synthetic):
    m = fit_interval(synthetic, ['x1', 'x2'])
    d = synthetic[np.isfinite(synthetic.R) & (synthetic.L > 0)].head(100)
    t = sample_times(m, d, d.L.to_numpy(), d.R.to_numpy(), np.random.default_rng(1))
    assert ((t >= d.L.to_numpy()) & (t <= d.R.to_numpy())).all()


def test_predict_dispatches_on_model_kind(synthetic):
    event = np.isfinite(synthetic.R.to_numpy())
    cox = fit_cox(synthetic, ['x1', 'x2'], np.where(event, synthetic.R, synthetic.L), event)
    grid = pd.DataFrame(dict(x1=np.linspace(-2, 2, 9), x2=0.0))
    p = predict(cox, grid, 730)
    assert cox['kind'] == 'cox' and ((p > 0) & (p < 1)).all()
    assert (np.diff(p) > 0).all()       # higher x1, higher risk


def test_null_model_gives_the_same_risk_to_everyone(synthetic):
    null = fit_interval(synthetic, [])
    p = predict(null, synthetic.head(10), 730)
    assert np.ptp(p) == 0 and 0 < p[0] < 1
