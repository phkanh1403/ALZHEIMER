"""Model inputs for a patient entered by hand, checked against the same rules the training pipeline applies.

Trends use the frozen training parameters (BLUP). Entries the models were never trained on are rejected with
InvalidEntry instead of being turned into a risk estimate:
  - history dates on or after the visit (the pipeline only uses data dated on or before the visit);
  - an MRI dated after the visit or more than MRI_MAX_AGE days before it;
  - earlier MRI scans at a different field strength from the current scan (trends use one series);
  - incomplete rows, an incomplete MRI, or values outside their valid range.
"""
import numpy as np
import pandas as pd

from .config import DAYS_PER_YEAR, MIN_SPAN, MRI_MAX_AGE
from .data import blup_slope

DATE = 'Date'
COG_HISTORY = {'date': DATE, 'adas13': 'ADAS-Cog13', 'faq': 'FAQ'}
MRI_HISTORY = {'date': DATE, 'hippo': 'Hippocampal volume (mm³)', 'icv': 'Intracranial volume (mm³)',
               'entor': 'Entorhinal thickness (mm)', 'field': 'Field strength'}
FIELD_STRENGTHS = ['3T', '1.5T']
BOUNDS = {
    'tau': (0.0, 20.0), 'age': (18.0, 115.0), 'sex': (1.0, 2.0), 'education': (0.0, 36.0),
    'mmse': (0.0, 30.0), 'faq': (0.0, 30.0), 'adas13': (0.0, 85.0), 'ravlt_delay': (0.0, 15.0),
    'hippo_adj': (1000.0, 15000.0), 'entor': (0.5, 6.0), 'field3t': (0.0, 1.0),
    'adas13_slope': (-30.0, 30.0), 'faq_slope': (-30.0, 30.0),
    'hippo_slope': (-2000.0, 2000.0), 'entor_slope': (-2.0, 2.0),
}


class InvalidEntry(ValueError):
    """The entered data cannot be used for a risk estimate; the message says why."""


def empty_cognitive_history():
    return pd.DataFrame({COG_HISTORY['date']: pd.Series(dtype='datetime64[ns]'),
                         COG_HISTORY['adas13']: pd.Series(dtype=float), COG_HISTORY['faq']: pd.Series(dtype=float)})


def empty_mri_history():
    return pd.DataFrame({MRI_HISTORY['date']: pd.Series(dtype='datetime64[ns]'),
                         **{MRI_HISTORY[k]: pd.Series(dtype=float) for k in ['hippo', 'icv', 'entor']},
                         MRI_HISTORY['field']: pd.Series(dtype=object)})


def _day(v):
    return str(pd.Timestamp(v).date())


def _series_slope(dates, values, params):
    """BLUP slope of a series; (None, span, start) when the series spans less than MIN_SPAN days."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    order = np.argsort(d.values)
    d, v = d.iloc[order].reset_index(drop=True), np.asarray(values, float)[order]
    span = int((d.iloc[-1] - d.iloc[0]).days)
    if span < MIN_SPAN:
        return None, span, d.iloc[0]
    t = ((d - d.iloc[0]).dt.days / DAYS_PER_YEAR).to_numpy()
    return blup_slope(t, v, params), span, d.iloc[0]


def _filled_rows(table, value_columns, what):
    """Rows with a date or a value (rows left empty are ignored); None when nothing was entered.

    Rows with values but no date are rejected.
    """
    if table is None or not len(table):
        return None
    table = table.dropna(how='all', subset=[DATE, *value_columns])
    if not len(table):
        return None
    if table[DATE].isna().any():
        raise InvalidEntry(f'{what}: every row with values needs a date.')
    return table.assign(**{DATE: pd.to_datetime(table[DATE])})


def _check_before(dates, limit, what, limit_name):
    late = sorted({_day(d) for d in dates if pd.Timestamp(d) >= limit})
    if late:
        raise InvalidEntry(f'{what} must be dated before the {limit_name} ({_day(limit)}); found {", ".join(late)}.')


def _current_mri(visit, mri_date, hv, icv, entor):
    given = [v is not None for v in (hv, icv, entor)]
    if any(given) and not all(given):
        raise InvalidEntry('Enter hippocampal volume, intracranial volume and entorhinal thickness together.')
    if not all(given):
        return None
    day = pd.Timestamp(mri_date) if mri_date else visit
    if day > visit:
        raise InvalidEntry(f'The MRI ({_day(day)}) is dated after the visit ({_day(visit)}).')
    if (visit - day).days > MRI_MAX_AGE:
        raise InvalidEntry(f'The MRI is {(visit - day).days} days older than the visit; the models only use an MRI '
                           f'from the {MRI_MAX_AGE} days before the visit. Leave the MRI fields empty instead.')
    return day


def _cognitive_trends(f, history, visit, adas13, faq, prep):
    history = _filled_rows(history, [COG_HISTORY['adas13'], COG_HISTORY['faq']], 'Earlier cognitive tests')
    if history is None:
        return
    _check_before(history[DATE], visit, 'Earlier cognitive tests', 'visit')
    for key, current in [('adas13', adas13), ('faq', faq)]:
        s = history.dropna(subset=[COG_HISTORY[key]])
        if len(s):
            slope, _, start = _series_slope([*s[DATE], visit], [*s[COG_HISTORY[key]], current], prep[key])
            if slope is not None:
                f[f'{key}_slope'], f[f'{key}_from'] = slope, _day(start)


def _atrophy_trends(f, history, mri_day, field, prep):
    cols = [MRI_HISTORY[k] for k in ['hippo', 'icv', 'entor']]
    history = _filled_rows(history, cols, 'Earlier MRI scans')
    if history is None:
        return
    if 'hippo_adj' not in f:
        raise InvalidEntry('Earlier MRI scans need a current MRI (hippocampal volume, intracranial volume and '
                           'entorhinal thickness).')
    if history[cols].isna().any(axis=None):
        raise InvalidEntry('Earlier MRI scans: fill in all three measurements on every row.')
    _check_before(history[DATE], mri_day, 'Earlier MRI scans', 'current MRI')
    fields = history.get(MRI_HISTORY['field'], pd.Series(None, index=history.index, dtype=object))
    other = history[fields != field]
    if len(other):
        dates = ', '.join(sorted({_day(d) for d in other[DATE]}))
        raise InvalidEntry(f'Earlier MRI scans must use the same field strength as the current scan ({field}); '
                           f'the rows dated {dates} have a different or missing field strength.')
    b, m = prep['icv']['b'], prep['icv']['icv_mean']
    dates = [*history[DATE], mri_day]
    adjusted = [*(history[cols[0]].astype(float) - b * (history[cols[1]].astype(float) - m)), f['hippo_adj']]
    slope, span, start = _series_slope(dates, adjusted, prep['hippo_adj'])
    f['mri_span_days'], f['mri_from'] = span, _day(start)
    if slope is not None:
        f['hippo_slope'] = slope
        f['entor_slope'] = _series_slope(dates, [*history[cols[2]].astype(float), f['entor']], prep['entor'])[0]


def _check_bounds(f):
    bad = [f'{k} = {f[k]:.3g} (valid {lo:g} to {hi:g})' for k, (lo, hi) in BOUNDS.items()
           if f.get(k) is not None and not lo <= f[k] <= hi]
    if bad:
        raise InvalidEntry('Values outside the range the models were trained on: ' + '; '.join(bad) + '.')


def manual_features(prep, *, visit, mci_years, age, sex, education, mmse, faq, adas13, ravlt, mri_date=None, hv=None,
                    icv=None, entor=None, field='3T', cog_history=None, mri_history=None):
    """Model inputs for one visit entered by hand. Raises InvalidEntry when the entry cannot be used."""
    visit = pd.Timestamp(visit)
    if field not in FIELD_STRENGTHS:
        raise InvalidEntry(f'Field strength must be one of {FIELD_STRENGTHS}.')
    f = dict(tau=float(mci_years), tau2=float(mci_years) ** 2, age=float(age), sex=1.0 if sex == 'Male' else 2.0,
             education=float(education), mmse=float(mmse), faq=float(faq), adas13=float(adas13),
             ravlt_delay=float(ravlt), mmse_date=_day(visit), adas13_date=_day(visit))
    mri_day = _current_mri(visit, mri_date, hv, icv, entor)
    if mri_day is not None:
        b, m = prep['icv']['b'], prep['icv']['icv_mean']
        f.update(hippo_adj=float(hv - b * (icv - m)), entor=float(entor), field3t=1.0 if field == '3T' else 0.0,
                 mri_date=_day(mri_day), mri_age_days=int((visit - mri_day).days))
    _cognitive_trends(f, cog_history, visit, adas13, faq, prep)
    _atrophy_trends(f, mri_history, mri_day, field, prep)
    _check_bounds(f)
    return f
