"""Status at a time point for interval-censored data, IPCW weights, the Turnbull estimator and evaluation metrics."""
from itertools import pairwise

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import (
    BAND_THRESHOLDS,
    CALIBRATION_GROUPS,
    DELTA_AUC,
    IPCW_MIN_SURVIVAL,
    NB_CORE,
    NB_MARGIN,
    THRESHOLDS,
    WINDOW,
)
from .model import cox_cumhaz, fit_cox, sample_times

CENSOR_COLS = ['tau', 'age', 'mmse', 'adas13', 'faq']
CATS = ['case_certain', 'control_certain', 'ambiguous', 'censored_early']
IMPUTE_MODES = ('ref', 'L', 'mid', 'R')
_EPS_DAY = 1e-6   # evaluate survival just before a day / place an onset just after L


def nb_key(threshold):
    """Column name for net benefit at a threshold, e.g. 0.15 -> 'nb_15'."""
    return f'nb_{round(threshold * 100)}'


def censoring_model(train):
    """Cox model for loss to follow-up before WINDOW (used for IPCW)."""
    event = np.isfinite(train.R.to_numpy())
    time = np.where(event, train.R, train.L)
    censored = (~event) & (train.L.to_numpy() < WINDOW - 1e-9)
    return fit_cox(train, CENSOR_COLS, time, censored)


def categories(L, R, h):
    """What is known at time point h: certain case, certain control, interval crossing h, or censored before h."""
    finite = np.isfinite(R)
    cat = np.full(len(L), 'censored_early', dtype=object)
    cat[finite & (R <= h)] = 'case_certain'
    cat[L >= h] = 'control_certain'
    cat[finite & (L < h) & (R > h)] = 'ambiguous'
    return cat


def status(frame, h, ref_model, rng, cmodel, mode='ref'):
    """Outcome y, IPCW weight and status group at time point h.

    mode decides the onset for intervals crossing h: 'ref' draws it from ref_model; 'L', 'mid' and 'R' place it
    just after L, at the midpoint or at R.
    """
    if mode not in IMPUTE_MODES:
        raise ValueError(f'mode must be one of {IMPUTE_MODES}, got {mode!r}')
    L, R = frame.L.to_numpy(float), frame.R.to_numpy(float)
    cat = categories(L, R, h)
    amb = cat == 'ambiguous'
    T = R.copy()
    if amb.any():
        if mode == 'ref':
            T[amb] = sample_times(ref_model, frame[amb], L[amb], R[amb], rng)
        elif mode == 'L':
            T[amb] = L[amb] + _EPS_DAY
        elif mode == 'mid':
            T[amb] = (L[amb] + R[amb]) / 2
    case = (cat == 'case_certain') | (amb & (T <= h))
    control = (cat == 'control_certain') | (amb & (T > h))
    Rc = np.where(np.isfinite(R), R, h)
    G_case = np.exp(-cox_cumhaz(cmodel, frame, np.maximum(Rc - _EPS_DAY, 0)))
    G_h = np.exp(-cox_cumhaz(cmodel, frame, np.full(len(frame), float(h))))
    w = np.zeros(len(frame))
    w[case] = 1 / np.maximum(G_case[case], IPCW_MIN_SURVIVAL)
    w[control] = 1 / np.maximum(G_h[control], IPCW_MIN_SURVIVAL)
    return case.astype(float), w, cat


# ----------------------------------------------------------------------- Turnbull (no IPCW needed)
def turnbull(L, R, tol=1e-8, iters=5000):
    """Turnbull NPMLE for (L, R]: innermost intervals (q, p] and their probability mass (self-consistency EM)."""
    L, R = np.asarray(L, float), np.asarray(R, float)
    points = sorted([(x, 1) for x in L] + [(x, 0) for x in R])   # ties: R (0) before L (1)
    inner = sorted({(a, b) for (a, ka), (b, kb) in pairwise(points) if ka == 1 and kb == 0})
    q = np.array([i[0] for i in inner])
    p = np.array([i[1] for i in inner])
    A = (q[None, :] >= L[:, None]) & (p[None, :] <= R[:, None])
    s = np.full(len(inner), 1 / len(inner))
    for _ in range(iters):
        den = A @ s
        new = (A * s / np.maximum(den[:, None], 1e-300)).mean(0)
        converged = np.abs(new - s).max() < tol
        s = new
        if converged:
            break
    return q, p, s


def turnbull_risk(L, R, h):
    """(lower, upper, midpoint) of the probability of onset by h."""
    q, p, s = turnbull(L, R)
    lower = s[p <= h].sum()            # mass certainly before h
    upper = s[q < h].sum()             # mass possibly before h
    return float(lower), float(upper), float((lower + upper) / 2)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def turnbull_calibration(L, R, p, h, groups=CALIBRATION_GROUPS, overall=None):
    """Observed (Turnbull) vs predicted risk by risk group; overall observed/predicted ratio and calibration slope."""
    L, R = np.asarray(L, float), np.asarray(R, float)
    group = np.minimum((np.argsort(np.argsort(p)) * groups) // len(p), groups - 1)
    obs = np.array([turnbull_risk(L[group == g], R[group == g], h)[2] for g in range(groups)])
    pred = np.array([p[group == g].mean() for g in range(groups)])
    overall = overall or turnbull_risk(L, R, h)
    slope = float(np.polyfit(_logit(pred), _logit(np.clip(obs, 0.01, 0.99)), 1)[0])
    return dict(oe=float(overall[2] / p.mean()), slope=slope, observed_turnbull=overall[2],
                turnbull_lower=overall[0], turnbull_upper=overall[1], pred=pred, obs=obs)


def net_benefit(y, w, p, t):
    """Weighted net benefit of acting when predicted risk >= t."""
    pos = p >= t
    return float(((w * y * pos).sum() - (w * (1 - y) * pos).sum() * t / (1 - t)) / w.sum())


# ----------------------------------------------------------------------- confusion matrix (IPCW-weighted)
CELLS = ('tp', 'fp', 'fn', 'tn')


def cm_key(cell, threshold):
    """Column name for a confusion-matrix cell at a threshold, e.g. ('tp', 0.15) -> 'cm_tp_15'."""
    return f'cm_{cell}_{round(threshold * 100)}'


def confusion(y, w, p, t):
    """Weighted shares of true/false positives/negatives when acting at predicted risk >= t.

    Each patient counts with its IPCW weight, so the shares estimate the confusion matrix without censoring.
    Shares sum to 1 over patients whose status is known (w > 0).
    """
    k = w > 0
    y, w, flagged = y[k], w[k], p[k] >= t
    total = w.sum()
    if total <= 0:
        return dict.fromkeys(CELLS, np.nan)
    case = y == 1
    return dict(tp=w[case & flagged].sum() / total, fp=w[~case & flagged].sum() / total,
                fn=w[case & ~flagged].sum() / total, tn=w[~case & ~flagged].sum() / total)


def _ratio(a, b):
    return float(a / b) if b > 0 else np.nan


def confusion_rates(tp, fp, fn, tn):
    """Sensitivity, specificity, PPV, NPV, F1 and the share flagged, from (weighted) confusion-matrix cells."""
    return dict(sensitivity=_ratio(tp, tp + fn), specificity=_ratio(tn, tn + fp), ppv=_ratio(tp, tp + fp),
                npv=_ratio(tn, tn + fn), f1=_ratio(2 * tp, 2 * tp + fp + fn), flagged=tp + fp)


def summarize(y, w, p, p_null, L=None, R=None, h=None, cat=None, calib=True, overall=None):
    """Discrimination (AUC, PR-AUC), overall accuracy, calibration, net benefit and the confusion matrix at the
    risk-band thresholds for one set of predictions."""
    k = w > 0
    out = dict(n=len(y), cases=float((y[k] == 1).sum()), controls=float((y[k] == 0).sum()))
    if cat is not None:
        for c in CATS:
            out[c] = int((cat == c).sum())
    out['auc'] = float(roc_auc_score(y[k], p[k], sample_weight=w[k])) if len(np.unique(y[k])) == 2 else np.nan
    out['brier'] = float((w * (y - p) ** 2).mean())
    out['brier_null'] = float((w * (y - p_null) ** 2).mean())
    out['ipa'] = 1 - out['brier'] / out['brier_null'] if out['brier_null'] > 0 else np.nan
    out['mean_pred'] = float(p.mean())
    out['observed_ipcw'] = float((w * y).sum() / w.sum()) if w.sum() > 0 else np.nan
    if calib and L is not None:
        c = turnbull_calibration(L, R, p, h, overall=overall)
        out.update(oe_turnbull=c['oe'], cal_slope_turnbull=c['slope'], observed_turnbull=c['observed_turnbull'])
    for t in THRESHOLDS:
        out[nb_key(t)] = net_benefit(y, w, p, t)
    out['nb_treat_all_20'] = net_benefit(y, w, np.ones_like(p), 0.20)
    out['nb_core'] = float(np.mean([out[nb_key(t)] for t in NB_CORE]))
    out['pr_auc'] = (float(average_precision_score(y[k], p[k], sample_weight=w[k]))
                     if len(np.unique(y[k])) == 2 else np.nan)
    for t in BAND_THRESHOLDS:
        out.update({cm_key(cell, t): v for cell, v in confusion(y, w, p, t).items()})
    return out


def decide(d_auc, lo_auc, hi_auc, d_nb, lo_nb, hi_nb):
    """Pre-registered decision rule for an added-value comparison."""
    if lo_auc > 0 and d_auc >= DELTA_AUC and lo_nb > 0:
        return 'Added value'
    if lo_auc > -DELTA_AUC and hi_auc < DELTA_AUC and hi_nb <= NB_MARGIN:
        return 'Equivalent'
    return 'Inconclusive'
