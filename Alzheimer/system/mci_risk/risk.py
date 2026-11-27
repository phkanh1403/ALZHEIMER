"""Patient-level risk logic used by the app: which models apply, the estimate with its range, and what drives it."""
import numpy as np
import pandas as pd

from .config import BAND_THRESHOLDS, MIN_SPAN, MODELS
from .model import predict, risk_curve

RISK_BINS = np.linspace(0, 1, 51)          # 2-percentage-point bins for the cohort risk histogram
RISK_LEVELS = np.linspace(0, 1, 201)       # quantile levels stored for percentile ranks

# Bands follow the range of decision thresholds used to evaluate the models (core net-benefit thresholds).
BAND_LOWER, BAND_UPPER = BAND_THRESHOLDS
BANDS = {'low': 'Lower risk', 'mid': 'Intermediate risk', 'high': 'Higher risk'}


def _missing(v):
    return v is None or bool(pd.isna(v))


def is_complete(row, model_name):
    return all(not _missing(row.get(c)) for c in MODELS[model_name])


def mri_trend_ok(row):
    span = row.get('mri_span_days')
    return not _missing(span) and span >= MIN_SPAN


def applicable_models(row):
    """Models whose inputs are all available, simplest first (the last one is the most complete)."""
    usable = [m for m in MODELS if is_complete(row, m)]
    if 'M3' in usable and not mri_trend_ok(row):
        usable.remove('M3')
    return usable


def risk_band(p):
    """'low', 'mid' or 'high'."""
    if p < BAND_LOWER:
        return 'low'
    return 'mid' if p < BAND_UPPER else 'high'


def features_frame(row, columns):
    return pd.DataFrame([{c: row[c] for c in columns}])


def estimate(model, draws, row, horizons):
    """Risk by each horizon and, when bootstrap draws exist, its 95% range: (risk, low, high) arrays."""
    frame = features_frame(row, model['columns'])
    risk = risk_curve(model, frame, horizons)[0]
    if not draws:
        return risk, None, None
    sims = np.array([risk_curve(d, frame, horizons)[0] for d in draws])
    low, high = np.quantile(sims, [.025, .975], axis=0)
    return risk, low, high


def contributions(model, row):
    """Each input's effect on the log hazard relative to an average training patient (tau and tau² combined)."""
    cols = model['columns']
    z = (np.array([row[c] for c in cols], float) - np.array(model['mean'])) / np.array(model['scale'])
    c = pd.Series(z * np.array(model['beta']), index=cols)
    if 'tau' in c and 'tau2' in c:
        c['tau'] += c.pop('tau2')
    return c.drop(labels=['field3t'], errors='ignore')


# --------------------------------------------------------------------------- cohort comparison (aggregated)
def risk_distribution(model, frame, horizons):
    """Aggregated distribution of predicted risk in a cohort: quantiles and a binned histogram per horizon.

    Only summary statistics are kept, so the result can be shared without participant-level data.
    """
    out = {}
    for h in horizons:
        p = predict(model, frame, h)
        out[str(h)] = dict(n=len(p), quantiles=np.quantile(p, RISK_LEVELS).round(6).tolist(),
                           counts=np.histogram(p, RISK_BINS)[0].tolist())
    return out


def share_below(distribution, p):
    """Share of the cohort with a lower predicted risk than p (percentile rank from the stored quantiles)."""
    q = np.asarray(distribution['quantiles'])
    return float(np.interp(p, q, RISK_LEVELS, left=0.0, right=1.0))
