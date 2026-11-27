"""Risk models for interval-censored time to dementia diagnosis.

Main model: proportional hazards with a piecewise-constant baseline hazard.
    S(t|z) = exp(-H0(t) exp(b'z)),   H0(t) = sum_k exp(a_k) * |[0, t] ∩ piece k|
    penalised log-likelihood:  sum_i w_i log[S(L_i|z_i) - S(R_i|z_i)] - lam/2 ||b||² - kappa/2 sum_k (a_k - a_{k-1})²
Also a ridge Cox model (Breslow ties) for the censoring distribution (IPCW) and sensitivity analysis S1.

Fitted models are plain JSON-serialisable dicts; `kind` says which predictor applies.
"""
import warnings

import numpy as np
from scipy.optimize import minimize

# six 182.5-day pieces to day 1,095, two 365-day pieces to 1,825, one open piece:
# keeps the right bound of intervals crossing day 1,095
CUTS = np.array([0, 182.5, 365, 547.5, 730, 912.5, 1095.0, 1460.0, 1825.0, np.inf])
_ETA_CLIP = 30.0      # bound on the linear predictor, keeps exp() finite
_TINY = 1e-300        # floor inside log(1 - exp(-x))


class ConvergenceWarning(UserWarning):
    """The optimiser stopped without meeting its convergence criterion."""


def exposure(t):
    """n x K matrix: time spent in each baseline piece over [0, t]."""
    t = np.asarray(t, float)
    if not np.isfinite(t).all():
        raise ValueError('exposure needs finite times')
    lo, hi = CUTS[:-1], CUTS[1:]
    return np.clip(t[:, None], lo, hi) - lo


def _design(frame, cols, mean=None, scale=None):
    """Standardised design matrix; mean and scale are estimated when not given."""
    X = frame[cols].to_numpy(float)
    if mean is None:
        mean, scale = X.mean(0), X.std(0)
        scale[scale == 0] = 1
    return (X - mean) / scale, np.asarray(mean), np.asarray(scale)


def person_weights(frame):
    """Each participant contributes total weight 1 across their visits (target population = people, not visits)."""
    if 'RID' not in frame:
        return np.ones(len(frame))
    return (1.0 / frame.groupby('RID').RID.transform('size')).to_numpy(float)


def _weights(frame, weights):
    if isinstance(weights, str):
        return person_weights(frame)
    return np.ones(len(frame)) if weights is None else np.asarray(weights, float)


def _check(opt, what, strict=True):
    """Return whether the optimiser converged.

    strict: raise on a hard failure and only warn when the line search stopped early (ABNORMAL);
    otherwise warn on any failure.
    """
    if opt.success:
        return True
    if not strict or 'ABNORMAL' in str(opt.message):
        warnings.warn(f'{what}: {opt.message}', ConvergenceWarning, stacklevel=3)
        return False
    raise RuntimeError(f'{what}: {opt.message}')


# --------------------------------------------------------------------------- interval-censored PH model
def interval_negloglik(theta, X, EL, ED, event, wt, lam, kappa):
    """Penalised negative log-likelihood and its gradient (theta = [log hazards a, coefficients b])."""
    K = EL.shape[1]
    a, b = theta[:K], theta[K:]
    ea = np.exp(a)
    u = np.exp(np.clip(X @ b, -_ETA_CLIP, _ETA_CLIP))
    A = EL @ ea                     # baseline cumulative hazard at L
    x = (ED @ ea) * u               # hazard accumulated over (L, R]
    one_minus_s = -np.expm1(-np.maximum(x, _TINY))                     # 1 - exp(-x), stable
    logterm = np.where(event, np.log(one_minus_s), 0.0)
    ratio = np.where(event, np.exp(-x) / one_minus_s, 0.0)
    ll = (wt * (-A * u + logterm)).sum()
    g_b = X.T @ (wt * (-A * u + ratio * x))
    g_a = ea * (-(EL * (wt * u)[:, None]).sum(0) + (ED * (wt * ratio * u)[:, None]).sum(0))
    diff = np.diff(a)
    pen = lam / 2 * b @ b + kappa / 2 * diff @ diff
    g_pen_a = np.zeros(K)
    g_pen_a[1:] += kappa * diff
    g_pen_a[:-1] -= kappa * diff
    return -(ll - pen), -np.r_[g_a - g_pen_a, g_b - lam * b]


def interval_problem(frame, cols, weights='person'):
    """Arrays the likelihood needs: design X, exposures at L and over (L, R], event flags and weights."""
    X, mean, scale = _design(frame, cols)
    L, R = frame.L.to_numpy(float), frame.R.to_numpy(float)
    event = np.isfinite(R)
    EL = exposure(L)
    ED = exposure(np.where(event, R, L)) - EL
    return dict(X=X, mean=mean, scale=scale, EL=EL, ED=ED, event=event, wt=_weights(frame, weights), L=L, R=R)


def fit_interval(frame, cols, lam=1.0, kappa=1.0, weights='person'):
    p = interval_problem(frame, cols, weights)
    K, wt, event = len(CUTS) - 1, p['wt'], p['event']
    rate = max((wt * event).sum(), 1e-3) / max(np.nansum(wt * np.where(event, p['R'], p['L'])), 1)
    start = np.r_[np.full(K, np.log(rate)), np.zeros(p['X'].shape[1])]
    opt = minimize(interval_negloglik, start, args=(p['X'], p['EL'], p['ED'], event, wt, lam, kappa),
                   jac=True, method='L-BFGS-B')
    converged = _check(opt, 'interval model')
    return dict(kind='pwexp_interval', columns=list(cols), mean=p['mean'].tolist(), scale=p['scale'].tolist(),
                log_hazard=opt.x[:K].tolist(), beta=opt.x[K:].tolist(), lam=lam,
                cuts=[float(c) if np.isfinite(c) else 'inf' for c in CUTS], n=len(frame),
                persons=int(frame.RID.nunique()) if 'RID' in frame else len(frame),
                events=int(event.sum()), loglik=float(-opt.fun), converged=converged)


def baseline_cumhaz(model, t):
    return exposure(np.atleast_1d(t)) @ np.exp(model['log_hazard'])


def linear_predictor(model, frame):
    if not model['columns']:
        return np.zeros(len(frame))
    X, _, _ = _design(frame, model['columns'], np.array(model['mean']), np.array(model['scale']))
    return X @ np.array(model['beta'])


def risk_curve(model, frame, horizons):
    """n x len(horizons) matrix: probability of a dementia diagnosis by each horizon (interval model)."""
    u = np.exp(linear_predictor(model, frame))
    return -np.expm1(-np.outer(u, baseline_cumhaz(model, horizons)))


def predict(model, frame, horizon):
    """Probability of a dementia diagnosis by `horizon` days after the visit, for either model kind."""
    if model.get('kind') == 'cox':
        return cox_predict(model, frame, horizon)
    u = np.exp(linear_predictor(model, frame))
    return -np.expm1(-baseline_cumhaz(model, horizon)[0] * u)


def sample_times(model, frame, L, R, rng):
    """Draw T within (L, R] from the model's conditional distribution (imputation for crossing intervals)."""
    u = np.exp(linear_predictor(model, frame))
    HL, HR = baseline_cumhaz(model, L) * u, baseline_cumhaz(model, R) * u
    target = HL - np.log1p(-rng.uniform(size=len(u)) * (-np.expm1(-(HR - HL)))) if len(u) else HL
    # invert the piecewise-linear H0
    H = target / u
    hazard = np.exp(np.array(model['log_hazard']))
    grid = np.r_[0, np.cumsum(hazard * np.diff(CUTS))]   # last element = inf
    k = np.clip(np.searchsorted(grid, H, side='right') - 1, 0, len(CUTS) - 2)
    t = CUTS[k] + (H - grid[k]) / hazard[k]
    return np.clip(t, L, R)


# --------------------------------------------------------------------------- Cox ridge (Breslow)
def fit_cox(frame, cols, time, event, penalty=1.0, weights='person'):
    """Ridge Cox with Breslow ties; per-participant weights by default for multi-visit data."""
    X, mean, scale = _design(frame, cols)
    t, e = np.asarray(time, float), np.asarray(event, bool)
    wt = _weights(frame, weights)
    times = np.unique(t[e])
    counts = np.array([wt[e & (t == u)].sum() for u in times])
    order = np.argsort(t)
    idx = np.searchsorted(t[order], times, side='left')    # first at-risk position for each event time
    Xe = (wt[e][:, None] * X[e]).sum(0)

    def objective(beta):
        z = X @ beta
        shift = z.max()
        w = wt * np.exp(z - shift)
        den = np.cumsum(w[order][::-1])[::-1][idx]
        num = np.cumsum((w[:, None] * X)[order][::-1], axis=0)[::-1][idx]
        loss = counts @ (np.log(den) + shift) - (wt[e] * z[e]).sum() + penalty * beta @ beta / 2
        grad = (counts[:, None] * num / den[:, None]).sum(0) - Xe + penalty * beta
        return loss, grad

    opt = minimize(objective, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B')
    converged = _check(opt, 'Cox model', strict=False)
    w = wt * np.exp(X @ opt.x)
    den = np.cumsum(w[order][::-1])[::-1][idx]
    return dict(kind='cox', columns=list(cols), mean=mean.tolist(), scale=scale.tolist(),
                beta=opt.x.tolist(), times=times.tolist(), baseline=np.cumsum(counts / den).tolist(),
                converged=converged)


def cox_cumhaz(model, frame, t):
    """H(t_i | x_i) where t may differ by row."""
    X, _, _ = _design(frame, model['columns'], np.array(model['mean']), np.array(model['scale']))
    idx = np.searchsorted(model['times'], np.asarray(t, float), side='right') - 1
    base = np.where(idx >= 0, np.r_[model['baseline'], 0][np.maximum(idx, 0)], 0.0)
    return base * np.exp(X @ np.array(model['beta']))


def cox_predict(model, frame, horizon):
    return -np.expm1(-cox_cumhaz(model, frame, np.full(len(frame), horizon)))
