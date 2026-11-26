"""Page: accuracy, calibration, decision curve, model trade-offs and whether repeat MRI adds value."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..config import CV_FOLDS, CV_REPEATS, DELTA_AUC, MODELS, PRIMARY, THRESHOLDS
from ..metrics import nb_key
from ..risk import BAND_LOWER, BAND_UPPER
from . import common as ui

SEC_FILES = [f'cv/SEC_{m}_metrics.csv' for m in MODELS] + [f'cv/SEC_{m}_info.json' for m in MODELS]
SENSITIVITY_SHOWN = ['M3 − M2', 'M3_twopoint − M2', 'M2_bsi − M2']
STATUS_LABELS = {
    'case_certain': 'Diagnosed before the time point',
    'ambiguous': 'Diagnosed at the first visit after it',
    'control_certain': 'Still dementia-free after it',
    'censored_early': 'Left follow-up before it',
}


def _performance_table():
    """One row per model and time frame: CV metrics of the available-data models plus held-out test AUC."""
    test = ui.run_csv('test_coverage.csv')
    rows = []
    for m in MODELS:
        cv = ui.run_csv(f'cv/SEC_{m}_metrics.csv')
        persons = ui.run_json(f'cv/SEC_{m}_info.json').get('persons')
        for _, r in cv[cv.scope == 'random'].iterrows():
            t = (test[(test.model == m) & (test.h == r.h) & (test.scope == 'random')] if test is not None
                 else pd.DataFrame())
            rows.append(dict(model=ui.MODEL_INFO[m][0], code=m, short=ui.SHORT[m], h=int(r.h), auc_cv=r.auc,
                             auc_test=t.auc.iloc[0] if len(t) else np.nan, oe=r.oe_turnbull, ipa=r.ipa,
                             nb=r.nb_core, persons=persons))
    return pd.DataFrame(rows)


def _headline(perf):
    best = perf[(perf.code == 'M3') & (perf.h == PRIMARY)].iloc[0]
    k = st.columns(4)
    k[0].metric(f'Accuracy at {ui.HORIZON_LABEL[PRIMARY]} (AUC)', f'{best.auc_cv:.2f}',
                help='Most complete model, cross-validation.')
    k[1].metric('Held-out test AUC', ui.fmt_value(best.auc_test, '{:.2f}'))
    k[2].metric('Predicted vs observed cases', f'{1 / best.oe:.2f}×',
                help='Ratio of predicted to observed diagnoses (1.00 = perfect).')
    k[3].metric('Patients used to build models', f'{int(perf.persons.max()):,}')


def _accuracy_tab(perf):
    a, b = st.columns(2)
    fig = px.line(perf, x='h', y='auc_cv', color='short', markers=True, color_discrete_sequence=ui.MODEL_COLORS,
                  labels={'h': 'Time frame (days)', 'auc_cv': 'AUC', 'short': ''})
    fig.update_layout(title='Accuracy by time frame (cross-validation)', height=380,
                      legend=dict(orientation='h', y=-0.25),
                      xaxis=dict(tickvals=list(ui.HORIZON_LABEL), ticktext=list(ui.HORIZON_LABEL.values())))
    ui.plot(fig, a)
    p2 = perf[perf.h == PRIMARY].melt(id_vars='short', value_vars=['auc_cv', 'auc_test'], var_name='set',
                                      value_name='AUC')
    p2['set'] = p2.set.map({'auc_cv': 'Cross-validation', 'auc_test': 'Held-out test'})
    fig = px.bar(p2, x='short', y='AUC', color='set', barmode='group', color_discrete_sequence=[ui.BLUE, ui.GREY],
                 text_auto='.2f')
    fig.update_layout(title=f'Cross-validation vs held-out test at {ui.HORIZON_LABEL[PRIMARY]}', height=380,
                      yaxis_range=[0.5, 1], legend=dict(orientation='h', y=-0.25, title=''), xaxis_title='')
    ui.plot(fig, b)


def _calibration_tab(perf):
    h = ui.horizon_radio(key='calh')
    a, b = st.columns([1.3, 1])
    cal = ui.run_csv('test_calibration.csv')
    if cal is None:
        a.info('Calibration on the held-out test set appears after running evaluate.py.')
    else:
        c = cal[cal.h == h].assign(model=lambda d: d.model.map(ui.SHORT))
        fig = px.line(c.sort_values(['model', 'predicted']), x='predicted', y='observed_turnbull', color='model',
                      markers=True, color_discrete_sequence=ui.MODEL_COLORS,
                      labels={'predicted': 'Predicted risk', 'observed_turnbull': 'Observed rate', 'model': ''})
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#777'),
                                 name='Perfect agreement'))
        fig.update_layout(title='Predicted vs observed (held-out test, 10 risk groups)', height=420,
                          xaxis=dict(range=[0, 1], tickformat='.0%'), yaxis=dict(range=[0, 1], tickformat='.0%'),
                          legend=dict(orientation='h', y=-0.2))
        ui.plot(fig, a)
    ratio = perf[perf.h == h][['short', 'oe']].assign(ratio=lambda d: 1 / d.oe)
    fig = px.bar(ratio, x='ratio', y='short', orientation='h', text_auto='.2f', color_discrete_sequence=[ui.BLUE])
    fig.add_vline(x=1, line_dash='dash', line_color='#555')
    fig.update_layout(title='Predicted ÷ observed diagnoses (cross-validation)', height=420,
                      xaxis_title='1.00 = perfect', yaxis_title='', xaxis_range=[0.7, 1.3])
    ui.plot(fig, b)
    st.caption('Observed rates use the Turnbull estimator, which accounts for dementia being recognised only at '
               'scheduled visits.')


def _decision_tab():
    main = ui.run_csv('cv/main_metrics.csv')
    if main is None:
        st.info('The decision curve appears after running validation.py (main analysis).')
        return
    m = main[(main.scope == 'random') & (main.h == PRIMARY)]
    percents = [round(t * 100) for t in THRESHOLDS]
    rows = [dict(model=ui.SHORT[r.model], threshold=pct, nb=r[nb_key(t)] * 100)
            for _, r in m.iterrows() for t, pct in zip(THRESHOLDS, percents, strict=True)]
    observed = float(m.observed_ipcw.iloc[0])
    for t, pct in zip(THRESHOLDS, percents, strict=True):
        treat_all = observed - (1 - observed) * t / (1 - t)
        rows.append(dict(model='Follow everyone closely', threshold=pct, nb=treat_all * 100))
        rows.append(dict(model='Follow no one closely', threshold=pct, nb=0.0))
    nb = pd.DataFrame(rows)
    fig = px.line(nb, x='threshold', y='nb', color='model', markers=True,
                  color_discrete_sequence=[*ui.MODEL_COLORS, ui.AMBER, '#333'],
                  labels={'threshold': 'Risk threshold for closer follow-up (%)',
                          'nb': 'Net benefit (per 100 patients)', 'model': ''})
    fig.add_vrect(x0=BAND_LOWER * 100, x1=BAND_UPPER * 100, fillcolor='#eef3fb', line_width=0, layer='below')
    fig.update_layout(title=f'Decision curve at {ui.HORIZON_LABEL[PRIMARY]}: correctly flagged patients per 100, '
                            'after penalising false alarms',
                      height=440, legend=dict(orientation='h', y=-0.25), yaxis_range=[-5, max(nb.nb) + 3])
    ui.plot(fig)
    st.caption('A model is useful where its line is above both "follow everyone" and "follow no one". The shaded band '
               f'({BAND_LOWER:.0%}–{BAND_UPPER:.0%}) is the range used to judge the models.')


def _profiles_tab(perf):
    p = perf[perf.h == PRIMARY]
    metrics = {'Accuracy (AUC)': 'auc_cv', 'Held-out AUC': 'auc_test', 'Explained risk (IPA)': 'ipa',
               'Net benefit': 'nb', 'Patients covered': 'persons'}
    fig = go.Figure()
    for _, r in p.iterrows():
        vals = []
        for col in metrics.values():
            lo, hi = p[col].min(), p[col].max()
            vals.append(0.3 + 0.7 * (r[col] - lo) / (hi - lo) if hi > lo else 1)
        fig.add_trace(go.Scatterpolar(r=[*vals, vals[0]], theta=[*metrics, next(iter(metrics))], fill='toself',
                                      name=r.short, opacity=0.5,
                                      line=dict(color=ui.MODEL_COLORS[list(MODELS).index(r.code)])))
    fig.update_layout(title=f'Trade-offs between models at {ui.HORIZON_LABEL[PRIMARY]} '
                            '(scaled: outer edge = best of the four)', height=480,
                      polar=dict(radialaxis=dict(visible=False, range=[0, 1])), legend=dict(orientation='h', y=-0.1))
    a, b = st.columns([1.3, 1])
    ui.plot(fig, a)
    table = p[['short', 'auc_cv', 'auc_test', 'ipa', 'nb', 'persons']].rename(columns={
        'short': 'Model', 'auc_cv': 'AUC (CV)', 'auc_test': 'AUC (test)', 'ipa': 'IPA', 'nb': 'Net benefit',
        'persons': 'Patients'})
    b.dataframe(table.round(3), hide_index=True, width='stretch')
    b.caption('Richer models are slightly more accurate but need more data, so they apply to fewer patients.')


def _verdict_table(summary):
    st.info(f"**Conclusion:** {summary['overall']}. {summary['test_note']}")
    rows = []
    for h, tier in summary['tiers'].items():
        for key, label in [('cv', 'Cross-validation'), ('test', 'Held-out test')]:
            x = tier[key]
            rows.append({'Time frame': ui.HORIZON_LABEL[int(h)], 'Data': label,
                         'AUC gain from atrophy trend': f"{ui.fmt_signed(x['d_auc'])} "
                                                        f"({ui.fmt_signed(x['auc_ci'][0])} to "
                                                        f"{ui.fmt_signed(x['auc_ci'][1])})",
                         'Net-benefit gain': f"{ui.fmt_signed(x['d_nb'])} ({ui.fmt_signed(x['nb_ci'][0])} to "
                                             f"{ui.fmt_signed(x['nb_ci'][1])})",
                         'Verdict': x['conclusion']})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    delta, margin = summary['protocol']['delta_auc'], summary['protocol']['nb_margin']
    st.caption(f"Verdict rule set before analysis: 'Added value' needs an AUC gain ≥ {delta} with a positive lower "
               f"bound and a positive net-benefit gain; 'Equivalent' needs the AUC interval within ±{delta} and the "
               f'net-benefit gain ≤ {margin}.')


def _sensitivity_forest(s):
    s = s[s.comparison.isin(SENSITIVITY_SHOWN)].copy()
    tag = s.analysis.str.extract(r'^(S\d+[a-z]?|L1)')[0].fillna('main')
    s = s[~(tag.isin(['S1', 'S2a']) & (s.comparison == 'M3 − M2'))]
    tag = tag[s.index]
    s['label'] = s.analysis.str.replace(r'^(S\d+[a-z]?|L1) · ', '', regex=True)
    s.loc[s.comparison == 'M3_twopoint − M2', 'label'] = 'Atrophy rate from only two scans'
    s.loc[(tag == 'S2b') & (s.comparison == 'M3 − M2'), 'label'] = 'ADNI-GO/2 subgroup'
    s.loc[s.comparison == 'M2_bsi − M2', 'label'] = 'ADNI-GO/2: boundary-shift integral instead'
    s['label'] = s.label.str.slice(0, 1).str.upper() + s.label.str.slice(1)
    s = s.iloc[::-1]
    fig = go.Figure(go.Scatter(x=s.d_auc, y=s.label, mode='markers', marker=dict(color=ui.BLUE, size=9),
                               error_x=dict(type='data', symmetric=False, array=s.auc_hi - s.d_auc,
                                            arrayminus=s.d_auc - s.auc_lo)))
    fig.add_vrect(x0=-DELTA_AUC, x1=DELTA_AUC, fillcolor='#eef3fb', line_width=0, layer='below')
    fig.add_vline(x=0, line_color='#555')
    fig.update_layout(title=f'AUC gain from MRI atrophy trend at {ui.HORIZON_LABEL[PRIMARY]}, across alternative '
                            'analyses', height=560,
                      xaxis_title=f'AUC gain (95% interval) · shaded = ±{DELTA_AUC} equivalence margin',
                      margin=dict(l=10, r=10, t=40, b=10))
    ui.plot(fig)


def _status_chart(main):
    m3 = main[(main.scope == 'random') & (main.model == 'M3')]
    status = m3.melt(id_vars='h', value_vars=list(STATUS_LABELS), var_name='status', value_name='patients')
    status['status'] = status.status.map(STATUS_LABELS)
    status['Time frame'] = status.h.map(ui.HORIZON_LABEL)
    fig = px.bar(status, x='patients', y='Time frame', color='status', orientation='h',
                 color_discrete_sequence=[ui.RED, '#e8a0a0', '#cfd8e3', '#7a869a'])
    fig.update_layout(title='What we know about each patient at each time point', height=300, barmode='stack',
                      legend=dict(orientation='h', y=-0.35, title=''), xaxis_title='Patients (average per repeat)')
    ui.plot(fig)


def _repeat_mri_tab():
    summary = ui.run_json('summary.json')
    if summary:
        _verdict_table(summary)
    else:
        st.info('The verdict appears after running evaluate.py and report.py.')
    sensitivity = ui.run_csv('sensitivity_summary.csv')
    if sensitivity is not None:
        _sensitivity_forest(sensitivity)
    main = ui.run_csv('cv/main_metrics.csv')
    if main is not None:
        _status_chart(main)


def render():
    st.title('How well do the risk estimates work?')
    st.caption(f'Results from repeated cross-validation on the training data ({CV_FOLDS} folds × {CV_REPEATS} '
               'repeats, one visit per patient) and from a separate held-out test set. AUC: how well the model ranks '
               'patients who are later diagnosed above those who are not (0.5 = chance, 1 = perfect).')
    if not ui.require(*SEC_FILES):
        return
    perf = _performance_table()
    _headline(perf)
    tabs = st.tabs(['Accuracy', 'Calibration', 'Clinical usefulness', 'Model profiles', 'Does repeat MRI help?'])
    with tabs[0]:
        _accuracy_tab(perf)
    with tabs[1]:
        _calibration_tab(perf)
    with tabs[2]:
        _decision_tab()
    with tabs[3]:
        _profiles_tab(perf)
    with tabs[4]:
        _repeat_mri_tab()
