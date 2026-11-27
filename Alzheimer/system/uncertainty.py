"""Participant-level bootstrap (200 draws) for individual risk intervals, for BOTH model sets.

bootstrap_paired.json     paired set (models.json): same visits with complete M3 inputs
bootstrap_secondary.json  available-data set (secondary_models.json): each model on its own complete visits

Limitation: features are computed once with the preprocessing parameters from all training data, so the intervals
reflect uncertainty in the risk-model coefficients and baseline hazard, not in the head-size/BLUP parameters.
"""
import numpy as np
import pandas as pd

from mci_risk.config import MODELS, RUNS_DIR
from mci_risk.files import load_json, save_json
from mci_risk.model import fit_interval

DRAW_KEYS = ['columns', 'mean', 'scale', 'log_hazard', 'beta']


def draws_for(frame, cols, lam, B, rng):
    """Refit the model on B participant-level bootstrap samples; returns the coefficients of each draw."""
    ids = frame.RID.unique()
    groups = dict(tuple(frame.groupby('RID')))
    out = []
    for _ in range(B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        # a participant drawn twice must count as two people for the per-participant weights
        sample = pd.concat([groups[i].assign(RID=f'{i}_{k}') for k, i in enumerate(pick)], ignore_index=True)
        m = fit_interval(sample, cols, lam=lam)
        out.append({k: m[k] for k in DRAW_KEYS})
    return out


def build(B=200, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(RUNS_DIR / 'train_landmarks.csv', parse_dates=['t0'])
    paired = load_json(RUNS_DIR / 'models.json')
    secondary = load_json(RUNS_DIR / 'secondary_models.json')
    shared = df.dropna(subset=MODELS['M3'])
    save_json({n: draws_for(shared, c, paired[n]['lam'], B, rng) for n, c in MODELS.items()},
              RUNS_DIR / 'bootstrap_paired.json')
    save_json({n: draws_for(df.dropna(subset=c), c, secondary[n]['lam'], B, rng) for n, c in MODELS.items()},
              RUNS_DIR / 'bootstrap_secondary.json')


if __name__ == '__main__':
    build()
