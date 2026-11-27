"""The pre-specified cross-validation analyses (main, leakage audit L1, sensitivity S1–S10, per-model SEC_*).

Each analysis is a declarative spec; validation.py runs them and report.py labels them.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .config import DAYS_PER_YEAR, MODEL_PAIRS, MODELS
from .cv import pair_label
from .model import fit_cox

S9_HV_CENTER = 6800      # mm³, about the cohort mean of hippo_adj; centres the volume x MRI-age interaction
S9_HV_SCALE = 1000       # mm³ per unit of the interaction term
_TREND_MODELS = {k: MODELS[k] for k in ['M2', 'M3']}


@dataclass(frozen=True)
class Analysis:
    """One cross-validation analysis.

    raw_options   keyword arguments for data.labelled_raw (confirmed, lag, series_key)
    transform     factory: Sources -> function selecting/deriving the analysis set from the feature frame
    extra         additional models {name: fit(train_frame)} scored alongside `models`
    reported      comparisons that report.py lists in the sensitivity summary (None: not reported)
    """
    label: str | None
    models: dict
    pairs: list = field(default_factory=lambda: [('M3', 'M2')])
    raw_options: dict = field(default_factory=dict)
    params: str = 'fold'
    transform: Callable | None = None
    extra: dict | None = None
    reported: tuple = (pair_label('M3', 'M2'),)


def _cox_m3(train):
    event = np.isfinite(train.R.to_numpy())
    return fit_cox(train, MODELS['M3'], np.where(event, train.R, train.L), event)


def _complete_m3(_src):
    return lambda frame: frame.dropna(subset=MODELS['M3'])


def _consistent_demographics(src):
    return lambda frame: frame[~frame.RID.isin(src.demo_conflict)].dropna(subset=MODELS['M3'])


def _mri_age_terms(_src):
    def transform(frame):
        frame = frame.dropna(subset=MODELS['M3']).copy()
        frame['hv_x_mriage'] = ((frame.hippo_adj - S9_HV_CENTER) / S9_HV_SCALE
                                * frame.mri_age_days / DAYS_PER_YEAR)
        frame['mri_age_y'] = frame.mri_age_days / DAYS_PER_YEAR
        return frame
    return transform


def _lag(days):
    return Analysis(f'S8 · MRI ≥ {days} day{"s" if days > 1 else ""} before visit', _TREND_MODELS,
                    raw_options=dict(lag=days))


ANALYSES: dict[str, Analysis] = {
    'main': Analysis('Primary analysis', MODELS, pairs=MODEL_PAIRS),
    'L1': Analysis('L1 · preprocessing fitted on all training data', _TREND_MODELS, params='global'),
    'S1': Analysis('S1 · standard Cox (visit date as event date)', _TREND_MODELS,
                   pairs=[('M3', 'M2'), ('M3_cox', 'M3')], extra={'M3_cox': _cox_m3},
                   reported=(pair_label('M3_cox', 'M3'), pair_label('M3', 'M2'))),
    'S2a': Analysis('S2a · two-scan atrophy rate',
                    {**_TREND_MODELS, 'M3_twopoint': MODELS['M2'] + ['hippo_rate2', 'entor_rate2']},
                    pairs=[('M3', 'M2'), ('M3_twopoint', 'M2')],
                    reported=(pair_label('M3_twopoint', 'M2'), pair_label('M3', 'M2'))),
    'S2b': Analysis('S2b · hippocampal BSI (ADNI-GO/2)', {**_TREND_MODELS, 'M2_bsi': MODELS['M2'] + ['hbsi_rate']},
                    pairs=[('M3', 'M2'), ('M2_bsi', 'M2')],
                    reported=(pair_label('M2_bsi', 'M2'), pair_label('M3', 'M2'))),
    'S4': Analysis('S4 · confirmed dementia only', _TREND_MODELS, raw_options=dict(confirmed=True)),
    'S5': Analysis('S5 · without FAQ',   # same participants as the main analysis, FAQ removed from predictors
                   {k: [c for c in cols if not c.startswith('faq')] for k, cols in _TREND_MODELS.items()},
                   transform=_complete_m3),
    'S7': Analysis('S7 · excluding inconsistent demographics', _TREND_MODELS, transform=_consistent_demographics),
    'S8_1': _lag(1),
    'S8_7': _lag(7),
    'S8_30': _lag(30),
    'S9': Analysis('S9 · MRI age interaction',
                   {'M1': MODELS['M1'], 'M1_age': MODELS['M1'] + ['mri_age_y', 'hv_x_mriage']},
                   pairs=[('M1_age', 'M1')], transform=_mri_age_terms, reported=(pair_label('M1_age', 'M1'),)),
    'S10': Analysis('S10 · same scanner model', _TREND_MODELS, raw_options=dict(series_key='version_model')),
    **{f'SEC_{k}': Analysis(None, {k: cols}, pairs=[], reported=()) for k, cols in MODELS.items()},
}
