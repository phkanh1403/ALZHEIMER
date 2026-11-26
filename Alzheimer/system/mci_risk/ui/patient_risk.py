"""Page: risk card, risk curve, cohort comparison, drivers, provenance and model comparison for one patient."""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..config import DAYS_PER_MONTH, HORIZONS, MIN_SPAN, MODELS
from ..model import predict
from ..risk import (
    BAND_LOWER,
    BAND_UPPER,
    BANDS,
    applicable_models,
    contributions,
    estimate,
    is_complete,
    mri_trend_ok,
    risk_band,
)
from . import common as ui

CURVE_DAYS = np.arange(30, max(HORIZONS) + 1, 15)
TOP_FACTORS = 8


def _select_patient(c1, c2, c3):
    """(row, visit, name, all visits of a cohort participant or None); None when there is nothing to show."""
    source = c1.radio('Patients from', ['My patients', 'ADNI study cohort'], horizontal=True, key='patient_source')
    if source == 'My patients':
        pf = ui.patient_store().all()
        if pf.empty:
            st.info('No patients saved yet. Use **Add patient** or import an ADNI export in **Patient list**.')
            return None
        ix = c2.selectbox('Patient', pf.index, format_func=lambda i: f"{pf.loc[i, 'name']} · visit {pf.loc[i, 'day']}")
        return json.loads(pf.loc[ix, 'features']), pf.loc[ix, 'day'], pf.loc[ix, 'name'], None
    inputs = ui.cohort_inputs()
    rid = c2.selectbox('Participant (RID)', sorted(inputs.RID.unique()), index=0)
    visits = inputs[inputs.RID == rid].sort_values('t0')
    dates = visits.t0.dt.date.astype(str).tolist()
    visit = c3.selectbox('MCI visit', dates, index=len(dates) - 1, help='Defaults to the most recent MCI visit.')
    row = visits[visits.t0.dt.date.astype(str) == visit].iloc[0].to_dict()
    return row, visit, f'RID {rid}', visits


def _risk_card(p, lo, hi, h):
    band = risk_band(p)
    color = ui.BAND_COLORS[band]
    interval = f'95% range {lo:.0%}–{hi:.0%}' if lo is not None else ''
    st.markdown(f"""<div class="card"><h4>Risk of dementia diagnosis within {ui.HORIZON_LABEL[h]}</h4>
    <div class="big" style="color:{color}">{p:.0%}</div>
    <div class="muted">{interval}</div>
    <div style="margin-top:8px"><span class="chip" style="background:{color}">{BANDS[band]}</span></div></div>""",
                unsafe_allow_html=True)
    lower, upper = BAND_LOWER * 100, BAND_UPPER * 100
    steps = [{'range': r, 'color': ui.BAND_FILL[b]} for r, b in
             [([0, lower], 'low'), ([lower, upper], 'mid'), ([upper, 100], 'high')]]
    gauge = go.Figure(go.Indicator(mode='gauge', value=p * 100, number={'suffix': '%'},
                                   gauge={'axis': {'range': [0, 100], 'ticksuffix': '%'},
                                          'bar': {'color': color, 'thickness': 0.3}, 'steps': steps}))
    gauge.update_layout(height=190, margin=dict(l=40, r=40, t=20, b=0))
    st.plotly_chart(gauge, width='stretch', config={'displayModeBar': False})
    st.caption(f'{ui.BAND_TEXT} — the range of decision thresholds used to evaluate the model; '
               'not a clinical guideline.')


def _risk_curve_chart(curve, low, high, h):
    months = CURVE_DAYS / DAYS_PER_MONTH
    fig = go.Figure()
    if low is not None:
        fig.add_trace(go.Scatter(x=np.r_[months, months[::-1]], y=np.r_[high, low[::-1]] * 100, fill='toself',
                                 fillcolor='rgba(31,95,191,0.15)', line=dict(width=0), hoverinfo='skip',
                                 name='95% range'))
    fig.add_trace(go.Scatter(x=months, y=curve * 100, line=dict(color=ui.BLUE, width=3), name='Estimated risk',
                             hovertemplate='%{x:.0f} months: %{y:.1f}%<extra></extra>'))
    fig.add_vline(x=h / DAYS_PER_MONTH, line_dash='dot', line_color=ui.GREY)
    fig.update_layout(title=f'Risk over the next {ui.HORIZON_LABEL[max(HORIZONS)]}', height=330,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title='Months from this visit',
                      yaxis_title='Cumulative risk (%)', showlegend=False, yaxis=dict(rangemode='tozero'))
    ui.plot(fig)


def _cohort_chart(model_name, p, h):
    cohort = ui.cohort_risk(model_name, 'available', h)
    share_below = float((cohort < p).mean() * 100)
    fig = px.histogram(x=cohort * 100, nbins=30, color_discrete_sequence=[ui.GREY])
    fig.add_vline(x=p * 100, line_color=ui.BAND_COLORS[risk_band(p)], line_width=3)
    fig.update_layout(title=f'Higher than {share_below:.0f}% of study patients', height=330,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title=f'Risk within {ui.HORIZON_LABEL[h]} (%)',
                      yaxis_title='Patients', bargap=0.05)
    ui.plot(fig)


def _drivers_chart(model, row):
    c = contributions(model, row)
    c = c.reindex(c.abs().sort_values(ascending=False).index[:TOP_FACTORS]).sort_values()
    names = [ui.FRIENDLY.get(k, k).split(' (')[0] for k in c.index]
    fig = go.Figure(go.Bar(x=c.values, y=names, orientation='h',
                           marker_color=[ui.RED if v > 0 else ui.GREEN for v in c.values],
                           hovertemplate='%{y}: %{x:+.2f}<extra></extra>'))
    fig.update_layout(title='What drives this estimate', height=360, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title='← lowers risk · compared with an average study patient · raises risk →')
    ui.plot(fig)
    st.caption(f'Top {TOP_FACTORS} factors. Bar length is the effect on the log hazard relative to an average study '
               'patient; the model also adjusts for MRI field strength.')


def _patient_tiles(row, model, model_name, name, visit):
    st.markdown('#### Patient at this visit')
    tiles = [('Age', 'age', '{:.0f}'), ('Education (y)', 'education', '{:.0f}'), ('MMSE /30', 'mmse', '{:.0f}'),
             ('FAQ /30', 'faq', '{:.0f}'), ('ADAS-Cog13 /85', 'adas13', '{:.1f}'),
             ('Delayed recall /15', 'ravlt_delay', '{:.0f}'), ('Hippocampus (mm³)', 'hippo_adj', '{:,.0f}'),
             ('Entorhinal (mm)', 'entor', '{:.2f}'), ('Hippocampus Δ (mm³/yr)', 'hippo_slope', '{:+.0f}')]
    cols = st.columns(3)
    for i, (label, key, pattern) in enumerate(tiles):
        cols[i % 3].metric(label, ui.fmt_value(row.get(key), pattern))
    sex = {1.0: 'Male', 2.0: 'Female'}.get(row.get('sex'), '—')
    st.caption(f'{name} · Sex: {sex} · Visit {visit}')
    scanner = row.get('scanner')
    with st.expander('Where these values come from'):
        st.markdown(
            f"- Latest MRI: **{ui.fmt_date(row.get('mri_date'))}**"
            + (f' · scanner {scanner}' if isinstance(scanner, str) and scanner != 'NA' else '')
            + f"\n- MRI trend uses scans from **{ui.fmt_date(row.get('mri_from'))}** "
              f"({row.get('mri_span_days') or '—'} days)"
            + f"\n- ADAS-Cog13 measured **{ui.fmt_date(row.get('adas13_date'))}**, trend from "
              f"**{ui.fmt_date(row.get('adas13_from'))}**"
            + f'\n- Model: **{ui.MODEL_INFO[model_name][0]}** — {ui.MODEL_INFO[model_name][1]}. '
              f"Trained on {model.get('persons_train', '—')} ADNI participants.")


def _trajectory(visits, h):
    """Clinical-model risk at each earlier MCI visit of a cohort participant."""
    if visits is None or len(visits) < 2:
        return
    m0 = ui.resources()['available']['M0']
    usable = visits[[is_complete(r, 'M0') for r in visits.to_dict('records')]]
    if len(usable) < 2:
        return
    t = pd.DataFrame(dict(visit=usable.t0, risk=predict(m0, usable, h) * 100))
    fig = px.line(t, x='visit', y='risk', markers=True, color_discrete_sequence=[ui.BLUE])
    fig.update_layout(title=f'Clinical-model risk at each MCI visit ({ui.HORIZON_LABEL[h]} time frame)', height=280,
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title='Risk (%)', xaxis_title='')
    ui.plot(fig)


def _compare_models(row, h):
    with st.expander('Compare all four models for this patient'):
        if not (is_complete(row, 'M3') and mri_trend_ok(row)):
            st.info('Needs all inputs including an MRI atrophy trend.')
            return
        res = ui.resources()
        rows = []
        for m in MODELS:
            risk, lo, hi = estimate(res['paired'][m], res['boot_paired'].get(m, []), row, [h])
            rows.append(dict(model=ui.MODEL_INFO[m][0], risk=risk[0] * 100,
                             lo=(lo[0] if lo is not None else risk[0]) * 100,
                             hi=(hi[0] if hi is not None else risk[0]) * 100))
        o = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=o.risk, y=o.model, orientation='h', marker_color=ui.MODEL_COLORS,
                               error_x=dict(type='data', symmetric=False, array=o.hi - o.risk,
                                            arrayminus=o.risk - o.lo)))
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=f'Risk within {ui.HORIZON_LABEL[h]} (%)')
        ui.plot(fig)
        st.caption('All four models here are trained on the same patients so they can be compared fairly. '
                   'Differences for a single patient are not evidence that one model is better — '
                   'see Model performance.')


def render():
    st.title('Patient risk assessment')
    selected = _select_patient(*st.columns([1.1, 1.6, 1.1]))
    if selected is None:
        return
    row, visit, name, visits = selected
    if st.toggle('Patient is receiving anti-amyloid antibody therapy'):
        st.error('Risk estimates are not available for patients on anti-amyloid antibodies: these drugs cause brain '
                 'volume loss that is unrelated to disease progression, so MRI-based estimates would be misleading.')
        return
    usable = applicable_models(row)
    if not usable:
        st.warning('This visit lacks the core clinical scores (MMSE, FAQ, ADAS-Cog13, delayed recall, age, sex, '
                   'education). Add them to obtain an estimate.')
        return
    b1, b2 = st.columns([2, 1])
    model_name = b1.selectbox('Model', usable[::-1], format_func=lambda m: ui.MODEL_INFO[m][0] + (
        ' — most complete for this patient' if m == usable[-1] else ''))
    h = ui.horizon_radio(b2)
    if 'M3' not in usable and is_complete(row, 'M2') and not mri_trend_ok(row):
        st.caption(f'ℹ️ Atrophy-trend model not shown: the MRI scans span less than {MIN_SPAN} days or are missing.')

    res = ui.resources()
    model = res['available'][model_name]
    draws = res['boot_available'].get(model_name, [])
    curve, low, high = estimate(model, draws, row, CURVE_DAYS)
    p, lo, hi = estimate(model, draws, row, [h])

    left, mid, right = st.columns([1.05, 1.5, 1.2])
    with left:
        _risk_card(p[0], lo[0] if lo is not None else None, hi[0] if hi is not None else None, h)
    with mid:
        _risk_curve_chart(curve, low, high, h)
    with right:
        _cohort_chart(model_name, p[0], h)
    d1, d2 = st.columns([1.3, 1])
    with d1:
        _drivers_chart(model, row)
    with d2:
        _patient_tiles(row, model, model_name, name, visit)
    _trajectory(visits, h)
    _compare_models(row, h)
