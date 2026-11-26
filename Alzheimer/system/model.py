"""Proportional hazards model with a piecewise-constant baseline hazard for interval-censored data.

S(t|z) = exp(-H0(t) exp(b'z)),  H0(t) = sum_k exp(a_k) * |[0,t] intersect piece k|
log-likelihood: sum log[S(L|z) - S(R|z)] - lam/2 ||b||^2 - kappa/2 sum (a_k - a_{k-1})^2
Also a ridge Cox model (Breslow) for the censoring model (IPCW) and sensitivity analysis S1.
"""
import numpy as np
from scipy.optimize import minimize

# six 182.5-day pieces to day 1,095, two 365-day pieces to 1,825, one open piece: keeps the right bound of intervals crossing day 1,095
CUTS = np.array([0, 182.5, 365, 547.5, 730, 912.5, 1095.0, 1460.0, 1825.0, np.inf])


def _exposure(t):
    """n x K matrix: time spent in each piece over [0, t]."""
    t = np.asarray(t, float)
    lo, hi = CUTS[:-1], CUTS[1:]
    if not np.isfinite(t).all():
        raise ValueError('exposure needs finite times')
    return np.clip(t[:, None], lo, hi) - lo


def _design(frame, cols, mean=None, scale=None):
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


def fit_interval(frame, cols, lam=1.0, kappa=1.0, weights='person'):
    X, mean, scale = _design(frame, cols)
    wt = person_weights(frame) if isinstance(weights, str) else (np.ones(len(frame)) if weights is None else np.asarray(weights, float))
    L, R = frame.L.to_numpy(float), frame.R.to_numpy(float)
    ev = np.isfinite(R)
    EL, ER = _exposure(L), _exposure(np.where(ev, R, L))
    ED = ER - EL
    K, p = len(CUTS) - 1, X.shape[1]

    def obj(theta):
        a, b = theta[:K], theta[K:]
        ea = np.exp(a)
        eta = np.clip(X @ b, -30, 30)
        u = np.exp(eta)
        A = EL @ ea
        Dh = ED @ ea
        x = Dh * u
        # numerically stable log(1 - exp(-x))
        logterm = np.where(ev, np.log(-np.expm1(-np.maximum(x, 1e-300))), 0.0)
        ratio = np.where(ev, np.exp(-x) / (-np.expm1(-np.maximum(x, 1e-300))), 0.0)
        ll = (wt * (-A * u + logterm)).sum()
        d_eta = wt * (-A * u + ratio * x)
        g_b = X.T @ d_eta
        g_a = ea * (-(EL * (wt * u)[:, None]).sum(0) + ((ED * (wt * ratio * u)[:, None]).sum(0)))
        diff = np.diff(a)
        pen = lam / 2 * b @ b + kappa / 2 * diff @ diff
        g_pen_a = np.zeros(K)
        g_pen_a[1:] += kappa * diff
        g_pen_a[:-1] -= kappa * diff
        return -(ll - pen), -np.r_[g_a - g_pen_a, g_b - lam * b]

    rate = max((wt * ev).sum(), 1e-3) / max(np.nansum(wt * np.where(ev, R, L)), 1)
    start = np.r_[np.full(K, np.log(rate)), np.zeros(p)]
    opt = minimize(obj, start, jac=True, method='L-BFGS-B')
    if not opt.success and 'ABNORMAL' not in str(opt.message):
        raise RuntimeError(opt.message)
    return dict(kind='pwexp_interval', columns=list(cols), mean=mean.tolist(), scale=scale.tolist(),
                log_hazard=opt.x[:K].tolist(), beta=opt.x[K:].tolist(), lam=lam,
                cuts=[float(c) if np.isfinite(c) else 'inf' for c in CUTS], n=int(len(frame)), persons=int(frame.RID.nunique()) if 'RID' in frame else int(len(frame)),
                events=int(ev.sum()), loglik=float(-opt.fun))


def _H0(model, t):
    return _exposure(np.atleast_1d(t)) @ np.exp(model['log_hazard'])


def linear_predictor(model, frame):
    if not model['columns']:
        return np.zeros(len(frame))
    X, _, _ = _design(frame, model['columns'], np.array(model['mean']), np.array(model['scale']))
    return X @ np.array(model['beta'])


def predict(model, frame, horizon):
    """Probability of a dementia diagnosis by `horizon` days after the visit."""
    u = np.exp(linear_predictor(model, frame))
    return -np.expm1(-_H0(model, horizon)[0] * u)


def sample_times(model, frame, L, R, rng):
    """Draw T within (L, R] from the model’s conditional distribution (imputation for crossing intervals)."""
    u = np.exp(linear_predictor(model, frame))
    HL, HR = _H0(model, L) * u, _H0(model, R) * u
    target = HL - np.log1p(-rng.uniform(size=len(u)) * (-np.expm1(-(HR - HL)))) if len(u) else HL
    # invert the piecewise-linear H0
    H = target / u
    grid = np.r_[0, np.cumsum(np.exp(model['log_hazard']) * np.diff(CUTS))]   # last element = inf
    k = np.clip(np.searchsorted(grid, H, side='right') - 1, 0, len(CUTS) - 2)
    t = CUTS[k] + (H - grid[k]) / np.exp(np.array(model['log_hazard']))[k]
    return np.clip(t, L, R)


# --------------------------------------------------------------------------- Cox ridge (Breslow)
def fit_cox(frame, cols, time, event, penalty=1.0, weights='person'):
    """Ridge Cox with Breslow ties; per-participant weights by default for multi-visit data."""
    X, mean, scale = _design(frame, cols)
    t, e = np.asarray(time, float), np.asarray(event, bool)
    wt = person_weights(frame) if isinstance(weights, str) else (np.ones(len(t)) if weights is None else np.asarray(weights, float))
    times = np.unique(t[e])
    counts = np.array([wt[e & (t == u)].sum() for u in times])
    order = np.argsort(t)
    ts = t[order]
    idx = np.searchsorted(ts, times, side='left')
    Xe = (wt[e][:, None] * X[e]).sum(0)

    def obj(beta):
        z = X @ beta
        sh = z.max()
        w = wt * np.exp(z - sh)
        cw = np.cumsum(w[order][::-1])[::-1]
        den = cw[idx]
        cx = np.cumsum((w[:, None] * X)[order][::-1], axis=0)[::-1]
        num = cx[idx]
        loss = counts @ (np.log(den) + sh) - (wt[e] * z[e]).sum() + penalty * beta @ beta / 2
        grad = (counts[:, None] * num / den[:, None]).sum(0) - Xe + penalty * beta
        return loss, grad

    opt = minimize(obj, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B')
    w = wt * np.exp(X @ opt.x)
    cw = np.cumsum(w[order][::-1])[::-1]
    den = cw[idx]
    return dict(kind='cox', columns=list(cols), mean=mean.tolist(), scale=scale.tolist(),
                beta=opt.x.tolist(), times=times.tolist(), baseline=np.cumsum(counts / den).tolist())


def cox_cumhaz(model, frame, t):
    """H(t_i | x_i) where t may differ by row."""
    X, _, _ = _design(frame, model['columns'], np.array(model['mean']), np.array(model['scale']))
    idx = np.searchsorted(model['times'], np.asarray(t, float), side='right') - 1
    base = np.where(idx >= 0, np.r_[model['baseline'], 0][np.maximum(idx, 0)], 0.0)
    return base * np.exp(X @ np.array(model['beta']))


def cox_predict(model, frame, horizon):
    return -np.expm1(-cox_cumhaz(model, frame, np.full(len(frame), horizon)))
