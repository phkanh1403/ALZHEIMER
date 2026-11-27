"""Plain-language descriptions of one patient's results, written from the numbers shown on screen.

Pure functions (no Streamlit), so every sentence the app shows can be tested.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

SMALL_CHANGE = 2.0     # percentage points; model-to-model differences below this are described as small
TREND_CHANGE = 5.0     # percentage points; a change in clinical risk across visits larger than this is a trend


def pct(p):
    """0.234 -> '23%'."""
    return f'{p * 100:.0f}%'


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return ''.join(items)
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def top_factors(contrib: pd.Series, labels: dict, k=2):
    """(names raising the risk, names lowering it): the k largest effects in each direction."""
    name = lambda c: labels.get(c, c).split(' (')[0]   # noqa: E731
    up = contrib[contrib > 0].sort_values(ascending=False).index[:k]
    down = contrib[contrib < 0].sort_values().index[:k]
    return [name(c) for c in up], [name(c) for c in down]


def band_rule(band, lower, upper):
    return {'low': f'below {pct(lower)}', 'mid': f'{pct(lower)}–{pct(upper)}', 'high': f'{pct(upper)} or more'}[band]


@dataclass(frozen=True)
class PatientResult:
    """What the risk page shows for one patient, one model and one time frame."""
    name: str
    visit: str
    model_label: str
    horizon_label: str
    risk: float
    low: float | None
    high: float | None
    band: str
    band_label: str
    share_below: float | None = None


def patient_summary(r: PatientResult, contrib: pd.Series, labels: dict, lower: float, upper: float):
    """One overview paragraph for the patient: risk, range, band, cohort position and main drivers."""
    text = (f'**{r.name} · visit {r.visit}:** estimated risk of a dementia diagnosis within {r.horizon_label} is '
            f'**{pct(r.risk)}**')
    if r.low is not None:
        text += f' (95% range {pct(r.low)}–{pct(r.high)})'
    text += f', in the **{r.band_label.lower()}** band ({band_rule(r.band, lower, upper)}).'
    if r.share_below is not None:
        text += f' This is higher than {r.share_below * 100:.0f}% of study patients.'
    up, down = top_factors(contrib, labels)
    if up:
        text += f' Raising the estimate most: {_join(up)}.'
    if down:
        text += f' Lowering it: {_join(down)}.'
    if r.low is not None and any(r.low < cut <= r.high for cut in (lower, upper)):
        text += ' The 95% range crosses a band boundary, so the band is uncertain.'
    return text + f' Model: {r.model_label}.'


def gauge_caption(risk, band_label, lower, upper):
    return (f'The needle marks {pct(risk)} on a 0–100% scale. Zones: green lower risk (below {pct(lower)}), amber '
            f'intermediate ({pct(lower)}–{pct(upper)}), red higher (≥ {pct(upper)}); this patient is in the '
            f'{band_label.lower()} zone. The cut-offs are the decision thresholds used to evaluate the models, '
            'not a clinical guideline.')


def cohort_caption(n, horizon_label, share_below):
    return (f'Grey bars: predicted risk within {horizon_label} for {n:,} study patients at their latest MCI visit, '
            f'from the same model. The coloured line is this patient: {share_below * 100:.0f}% of study patients '
            'have a lower estimate.')


def drivers_caption(contrib: pd.Series, labels: dict, shown: int):
    up, down = top_factors(contrib, labels, k=1)
    text = (f'Top {shown} factors, compared with an average study patient: red bars raise the estimate, green bars '
            'lower it; bar length is the effect on the log hazard.')
    if up and down:
        text += f' The largest push upwards comes from {up[0]}, the largest push downwards from {down[0]}.'
    elif up or down:
        text += f' Every factor shown pushes the estimate {"upwards" if up else "downwards"}.'
    return text + ' The model also adjusts for MRI field strength.'


def trajectory_caption(risks: Sequence[float], horizon_label):
    first, last, n = risks[0] * 100, risks[-1] * 100, len(risks)
    change = last - first
    if abs(change) < TREND_CHANGE:
        trend = f'stayed about the same ({first:.0f}% → {last:.0f}%)'
    else:
        trend = f'{"rose" if change > 0 else "fell"} from {first:.0f}% to {last:.0f}%'
    return (f'Each point is the clinical-model risk within {horizon_label}, computed from the data available at one '
            f'of this participant\'s {n} MCI visits. Across these visits the risk {trend}.')


@dataclass(frozen=True)
class ModelEstimate:
    label: str
    risk: float
    low: float
    high: float
    band: str


def comparison_note(estimates: Sequence[ModelEstimate], band_labels: dict):
    """Explain a four-model comparison for one patient: the step with the largest change and band agreement."""
    risks = np.array([e.risk for e in estimates]) * 100
    steps = np.diff(risks)
    lines = ['Estimates: ' + '; '.join(f'{e.label} {pct(e.risk)}' for e in estimates) + '.']
    k = int(np.argmax(np.abs(steps)))
    if abs(steps[k]) < SMALL_CHANGE:
        lines.append(f'Adding MRI and trend data changes the estimate by less than {SMALL_CHANGE:.0f} percentage '
                     'points at every step, so the extra data add little for this patient.')
    else:
        verb = 'raises' if steps[k] > 0 else 'lowers'
        lines.append(f'The largest change comes from moving to **{estimates[k + 1].label}**, which {verb} the '
                     f'estimate by {abs(steps[k]):.0f} percentage points.')
    bands = [e.band for e in estimates]
    if len(set(bands)) == 1:
        lines.append(f'All four models place this patient in the {band_labels[bands[0]].lower()} band.')
    else:
        lines.append('The models disagree on the band (' + ', '.join(
            f'{e.label}: {band_labels[e.band].lower()}' for e in estimates) + '), so the band depends on which '
            'data are used.')
    if max(e.low for e in estimates) <= min(e.high for e in estimates):
        lines.append('The 95% ranges of all four models overlap, so the differences are within the uncertainty '
                     'of the estimates.')
    lines.append('All four models are trained on the same patients so they can be compared fairly; differences for '
                 'one patient are not evidence that one model is better (see Model performance).')
    return ' '.join(lines)
