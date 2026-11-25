"""Trạng thái tại chân trời h với dữ liệu kiểm duyệt khoảng, IPCW và các chỉ số đánh giá."""
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from model import fit_cox, cox_cumhaz, sample_times

CENSOR_COLS = ['tau', 'age', 'mmse', 'adas13', 'faq']
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NB_CORE = [0.15, 0.20, 0.25, 0.30]


def censoring_model(train):
    """Cox cho thời gian kiểm duyệt (bỏ khám) có hiệp biến; cắt hành chính không tính là biến cố."""
    ev = np.isfinite(train.R.to_numpy())
    time = np.where(ev, train.R, train.L)
    cens = (~ev) & (train.L.to_numpy() < 1095 - 1e-9)
    return fit_cox(train, CENSOR_COLS, time, cens)


def status(frame, h, ref_model, rng, cmodel):
    """Trả về y (0/1) và trọng số IPCW; ca bất định L<h<R được rút T từ mô hình tham chiếu M0."""
    L, R = frame.L.to_numpy(float), frame.R.to_numpy(float)
    T = R.copy()
    amb = (L < h) & (R > h) & np.isfinite(R)
    if amb.any():
        T[amb] = sample_times(ref_model, frame[amb], L[amb], R[amb], rng)
    case = np.isfinite(R) & (T <= h)
    control = ~case & ((L >= h) | (np.isfinite(R) & (T > h)))
    G_case = np.exp(-cox_cumhaz(cmodel, frame, np.maximum(R - 1e-6, 0)))
    G_h = np.exp(-cox_cumhaz(cmodel, frame, np.full(len(frame), float(h))))
    w = np.zeros(len(frame))
    w[case] = 1 / np.maximum(G_case[case], 0.05)
    w[control] = 1 / np.maximum(G_h[control], 0.05)
    return case.astype(float), w


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration(y, w, p):
    k = w > 0
    y, w, lp = y[k], w[k], _logit(p[k])

    def nll(theta, slope_free):
        a, b = theta[0], (theta[1] if slope_free else 1.0)
        eta = a + b * lp
        return -(w * (y * eta - np.logaddexp(0, eta))).sum()
    intercept = minimize(lambda t: nll(t, False), [0.0]).x[0]
    slope = minimize(lambda t: nll(t, True), [0.0, 1.0]).x[1]
    return float(intercept), float(slope)


def net_benefit(y, w, p, t):
    n = w.sum()
    pos = p >= t
    return float(((w * y * pos).sum() - (w * (1 - y) * pos).sum() * t / (1 - t)) / n)


def km_risk(T, E, h):
    T, E = np.asarray(T, float), np.asarray(E, int)
    S = 1.0
    for t in np.unique(T[E == 1]):
        if t > h:
            break
        S *= 1 - ((T == t) & (E == 1)).sum() / (T >= t).sum()
    return 1 - S


def km_calibration(L, R, p, h, groups=5):
    """Hiệu chuẩn không phụ thuộc mô hình: Kaplan–Meier (biến cố tại điểm giữa khoảng) theo nhóm nguy cơ."""
    L, R = np.asarray(L, float), np.asarray(R, float)
    E = np.isfinite(R).astype(int)
    T = np.where(E == 1, (L + np.where(E == 1, R, 0)) / 2, L)
    q = np.minimum((np.argsort(np.argsort(p)) * groups) // len(p), groups - 1)
    obs = np.array([km_risk(T[q == g], E[q == g], h) for g in range(groups)])
    pred = np.array([p[q == g].mean() for g in range(groups)])
    oe = km_risk(T, E, h) / p.mean()
    x, yv = _logit(pred), _logit(np.clip(obs, 0.01, 0.99))
    slope = float(np.polyfit(x, yv, 1)[0])
    return float(oe), slope, pred, obs


def summarize(y, w, p, p_null, L=None, R=None, h=None):
    k = w > 0
    out = dict(n=int(len(y)), cases=int((y[k] == 1).sum()), controls=int((y[k] == 0).sum()),
               unknown=int((~k).sum()))
    out['auc'] = float(roc_auc_score(y[k], p[k], sample_weight=w[k])) if len(np.unique(y[k])) == 2 else np.nan
    out['brier'] = float((w * (y - p) ** 2).mean())
    brier_null = float((w * (y - p_null) ** 2).mean())
    out['brier_null'] = brier_null
    out['ipa'] = 1 - out['brier'] / brier_null if brier_null > 0 else np.nan
    out['mean_pred'] = float(p.mean())
    out['observed'] = float((w * y).sum() / w.sum()) if w.sum() > 0 else np.nan
    if L is not None:
        out['oe_km'], out['cal_slope_km'], _, _ = km_calibration(L, R, p, h)
    for t in THRESHOLDS:
        out[f'nb_{int(round(t * 100))}'] = net_benefit(y, w, p, t)
    out['nb_treat_all_20'] = net_benefit(y, w, np.ones_like(p), 0.20)
    out['nb_core'] = float(np.mean([out[f'nb_{int(round(t * 100))}'] for t in NB_CORE]))
    return out


def decide(d_auc, lo_auc, hi_auc, d_nb, lo_nb, hi_nb, delta=0.02, nb_margin=0.01):
    if lo_auc > 0 and d_auc >= delta and lo_nb > 0:
        return 'Có giá trị gia tăng'
    if lo_auc > -delta and hi_auc < delta and hi_nb <= nb_margin:
        return 'Tương đương (MRI lặp không cần cho tiên lượng)'
    return 'Bất định'
