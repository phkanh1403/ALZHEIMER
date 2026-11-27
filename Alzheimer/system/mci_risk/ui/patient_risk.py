"""Page: overview, risk card, cohort comparison, drivers, provenance and model comparison for one patient."""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..config import MIN_SPAN, MODELS
from ..model import predict
from ..narrative import (
    ModelEstimate,
    PatientResult,
    cohort_caption,
    comparison_note,
    drivers_caption,
    gauge_caption,
    patient_summary,
    trajectory_caption,
)
from ..risk import (
    BAND_LOWER,
    BAND_UPPER,
    BANDS,
    RISK_BINS,
    applicable_models,
    contributions,
    estimate,
    is_complete,
    mri_trend_ok,
    risk_band,
    share_below,
)
from . import common as ui

TOP_FACTORS = 8
SOURCES = ['My patients', 'ADNI study cohort']


def _select_patient(c1, c2, c3):
    """(row, visit, name, all visits of a cohort participant or None); None when there is nothing to show."""
    sources = SOURCES if ui.has_cohort_data() else SOURCES[:1]
    source = c1.radio('Patients from', sources, horizontal=True, key='patient_source')
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
    st.caption(gauge_caption(p, BANDS[band], BAND_LOWER, BAND_UPPER))


def _cohort_chart(dist, p, h, share):
    if dist is None:
        st.info('The comparison with study patients appears after running train.py (cohort_risk.json).')
        return
    centres = (RISK_BINS[:-1] + RISK_BINS[1:]) / 2 * 100
    fig = go.Figure(go.Bar(x=centres, y=dist['counts'], marker_color=ui.GREY, width=1.9,
                           hovertemplate='%{x:.0f}%: %{y} patients<extra></extra>'))
    fig.add_vline(x=p * 100, line_color=ui.BAND_COLORS[risk_band(p)], line_width=3)
    fig.update_layout(title=f'Higher than {share * 100:.0f}% of study patients', height=330,
                      margin=dict(l=10, r=10, t=40, b=10), xaxis_title=f'Risk within {ui.HORIZON_LABEL[h]} (%)',
                      yaxis_title='Patients', xaxis_range=[0, 100])
    ui.plot(fig)
    st.caption(cohort_caption(dist['n'], ui.HORIZON_LABEL[h], share))


def _drivers_chart(contrib):
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:TOP_FACTORS]).sort_values()
    names = [ui.FRIENDLY.get(k, k).split(' (')[0] for k in c.index]
    fig = go.Figure(go.Bar(x=c.values, y=names, orientation='h',
                           marker_color=[ui.RED if v > 0 else ui.GREEN for v in c.values],
                           hovertemplate='%{y}: %{x:+.2f}<extra></extra>'))
    fig.update_layout(title='What drives this estimate', height=360, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title='← lowers risk · compared with an average study patient · raises risk →')
    ui.plot(fig)
    st.caption(drivers_caption(c, ui.FRIENDLY, len(c)))


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
    st.caption(f'{name} · Sex: {sex} · Visit {visit}. Hippocampal volume is adjusted for head size; '
               '"—" means the value is not available at this visit.')
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
    risk = predict(m0, usable, h)
    t = pd.DataFrame(dict(visit=usable.t0, risk=risk * 100))
    fig = px.line(t, x='visit', y='risk', markers=True, color_discrete_sequence=[ui.BLUE])
    fig.update_layout(title=f'Clinical-model risk at each MCI visit ({ui.HORIZON_LABEL[h]} time frame)', height=280,
                      margin=dict(l=10, r=10, t=40, b=10), yaxis_title='Risk (%)', xaxis_title='')
    ui.plot(fig)
    st.caption(trajectory_caption(list(risk), ui.HORIZON_LABEL[h]))


def _compare_models(row, h):
    with st.expander('Compare all four models for this patient'):
        if not (is_complete(row, 'M3') and mri_trend_ok(row)):
            st.info('Needs all inputs including an MRI atrophy trend.')
            return
        res = ui.resources()
        estimates = []
        for m in MODELS:
            risk, lo, hi = estimate(res['paired'][m], res['boot_paired'].get(m, []), row, [h])
            r = float(risk[0])
            estimates.append(ModelEstimate(ui.MODEL_INFO[m][0], r, float(lo[0]) if lo is not None else r,
                                           float(hi[0]) if hi is not None else r, risk_band(r)))
        o = pd.DataFrame([e.__dict__ for e in estimates])
        fig = go.Figure(go.Bar(x=o.risk * 100, y=o.label, orientation='h', marker_color=ui.MODEL_COLORS,
                               error_x=dict(type='data', symmetric=False, array=(o.high - o.risk) * 100,
                                            arrayminus=(o.risk - o.low) * 100)))
        for cut in (BAND_LOWER, BAND_UPPER):
            fig.add_vline(x=cut * 100, line_dash='dot', line_color=ui.GREY)
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=f'Risk within {ui.HORIZON_LABEL[h]} (%) · dotted lines: band cut-offs')
        ui.plot(fig)
        st.markdown(comparison_note(estimates, BANDS))


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
    risk, low, high = estimate(model, res['boot_available'].get(model_name, []), row, [h])
    p = float(risk[0])
    lo, hi = (float(low[0]), float(high[0])) if low is not None else (None, None)
    contrib = contributions(model, row)
    dist = ui.cohort_distribution(model_name, h)
    share = share_below(dist, p) if dist else None
    band = risk_band(p)
    result = PatientResult(name=name, visit=str(visit), model_label=ui.MODEL_INFO[model_name][0],
                           horizon_label=ui.HORIZON_LABEL[h], risk=p, low=lo, high=hi, band=band,
                           band_label=BANDS[band], share_below=share)
    with st.container(border=True):
        st.markdown(patient_summary(result, contrib, ui.FRIENDLY, BAND_LOWER, BAND_UPPER))

    left, right = st.columns([1, 1.4])
    with left:
        _risk_card(p, lo, hi, h)
    with right:
        _cohort_chart(dist, p, h, share)
    d1, d2 = st.columns([1.3, 1])
    with d1:
        _drivers_chart(contrib)
    with d2:
        _patient_tiles(row, model, model_name, name, visit)
    _trajectory(visits, h)
    _compare_models(row, h)

