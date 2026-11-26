"""Status at time point h for interval-censored data, IPCW, Turnbull estimator and evaluation metrics."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from model import fit_cox, cox_cumhaz, sample_times
from pipeline import ROOT, WINDOW

PROTOCOL = json.loads((ROOT / 'runs' / 'protocol.json').read_text(encoding='utf8'))
CENSOR_COLS = ['tau', 'age', 'mmse', 'adas13', 'faq']
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
NB_CORE = PROTOCOL['nb_core_thresholds']
CATS = ['case_certain', 'control_certain', 'ambiguous', 'censored_early']


def censoring_model(train):
    ev = np.isfinite(train.R.to_numpy())
    time = np.where(ev, train.R, train.L)
    cens = (~ev) & (train.L.to_numpy() < WINDOW - 1e-9)
    return fit_cox(train, CENSOR_COLS, time, cens)


def categories(L, R, h):
    fin = np.isfinite(R)
    cat = np.full(len(L), 'censored_early', dtype=object)
    cat[fin & (R <= h)] = 'case_certain'
    cat[L >= h] = 'control_certain'
    cat[fin & (L < h) & (R > h)] = 'ambiguous'
    return cat


def status(frame, h, ref_model, rng, cmodel, mode='ref'):
    """Outcome y, IPCW weight and status group. mode: ref (onset drawn from the reference model), L, mid, R."""
    L, R = frame.L.to_numpy(float), frame.R.to_numpy(float)
    cat = categories(L, R, h)
    amb = cat == 'ambiguous'
    T = R.copy()
    if amb.any():
        if mode == 'ref':
            T[amb] = sample_times(ref_model, frame[amb], L[amb], R[amb], rng)
        elif mode == 'L':
            T[amb] = L[amb] + 1e-6
        elif mode == 'mid':
            T[amb] = (L[amb] + R[amb]) / 2
        elif mode == 'R':
            T[amb] = R[amb]
    case = (cat == 'case_certain') | (amb & (T <= h))
    control = (cat == 'control_certain') | (amb & (T > h))
    Rc = np.where(np.isfinite(R), R, h)
    G_case = np.exp(-cox_cumhaz(cmodel, frame, np.maximum(Rc - 1e-6, 0)))
    G_h = np.exp(-cox_cumhaz(cmodel, frame, np.full(len(frame), float(h))))
    w = np.zeros(len(frame))
    w[case] = 1 / np.maximum(G_case[case], 0.05)
    w[control] = 1 / np.maximum(G_h[control], 0.05)
    return case.astype(float), w, cat


# ----------------------------------------------------------------------- Turnbull (no IPCW needed)
def turnbull(L, R, tol=1e-8, iters=5000):
    """Turnbull NPMLE for (L, R]; returns innermost intervals (q, p] and their mass."""
    L, R = np.asarray(L, float), np.asarray(R, float)
    pts = sorted([(x, 1) for x in L] + [(x, 0) for x in R])   # ties: R (0) before L (1)
    inner = []
    for (a, ka), (b, kb) in zip(pts[:-1], pts[1:]):
        if ka == 1 and kb == 0:
            inner.append((a, b))
    inner = sorted(set(inner))
    q = np.array([i[0] for i in inner]); p = np.array([i[1] for i in inner])
    A = (q[None, :] >= L[:, None]) & (p[None, :] <= R[:, None])
    s = np.full(len(inner), 1 / len(inner))
    for _ in range(iters):
        den = A @ s
        new = (A * s / np.maximum(den[:, None], 1e-300)).mean(0)
        if np.abs(new - s).max() < tol:
            s = new; break
        s = new
    return q, p, s


def turnbull_risk(L, R, h):
    q, p, s = turnbull(L, R)
    lower = s[p <= h].sum()            # mass certainly before h
    upper = s[q < h].sum()             # mass possibly before h
    return float(lower), float(upper), float((lower + upper) / 2)


def turnbull_calibration(L, R, p, h, groups=10, overall=None):
    L, R = np.asarray(L, float), np.asarray(R, float)
    q = np.minimum((np.argsort(np.argsort(p)) * groups) // len(p), groups - 1)
    obs = np.array([turnbull_risk(L[q == g], R[q == g], h)[2] for g in range(groups)])
    pred = np.array([p[q == g].mean() for g in range(groups)])
    overall = overall or turnbull_risk(L, R, h)
    oe = overall[2] / p.mean()
    x, yv = _logit(pred), _logit(np.clip(obs, 0.01, 0.99))
    slope = float(np.polyfit(x, yv, 1)[0])
    return dict(oe=float(oe), slope=slope, observed_turnbull=overall[2], turnbull_lower=overall[0],
                turnbull_upper=overall[1], pred=pred, obs=obs)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def net_benefit(y, w, p, t):
    n = w.sum()
    pos = p >= t
    return float(((w * y * pos).sum() - (w * (1 - y) * pos).sum() * t / (1 - t)) / n)


def summarize(y, w, p, p_null, L=None, R=None, h=None, cat=None, calib=True, overall=None):
    k = w > 0
    out = dict(n=int(len(y)), cases=float((y[k] == 1).sum()), controls=float((y[k] == 0).sum()))
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
        out[f'nb_{int(round(t * 100))}'] = net_benefit(y, w, p, t)
    out['nb_treat_all_20'] = net_benefit(y, w, np.ones_like(p), 0.20)
    out['nb_core'] = float(np.mean([out[f'nb_{int(round(t * 100))}'] for t in NB_CORE]))
    return out


def decide(d_auc, lo_auc, hi_auc, d_nb, lo_nb, hi_nb):
    delta, margin = PROTOCOL['delta_auc'], PROTOCOL['nb_margin']
    if lo_auc > 0 and d_auc >= delta and lo_nb > 0:
        return 'Added value'
    if lo_auc > -delta and hi_auc < delta and hi_nb <= margin:
        return 'Equivalent'
    return 'Inconclusive'
