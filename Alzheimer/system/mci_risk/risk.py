"""Patient-level risk logic used by the app: which models apply, the estimate with its range, and what drives it."""
import numpy as np
import pandas as pd

from .config import MIN_SPAN, MODELS, NB_CORE
from .model import risk_curve

# Bands follow the range of decision thresholds used to evaluate the models (core net-benefit thresholds).
BAND_LOWER, BAND_UPPER = min(NB_CORE), max(NB_CORE)
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
