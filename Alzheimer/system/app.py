"""MCI Dementia Risk — clinical decision-support app for patients with mild cognitive impairment."""
import json
import sqlite3
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pipeline import ROOT, MODELS, LABELS, MIN_SPAN, blup_slope, read
from model import _exposure, linear_predictor
from registry import initialize, import_zip, FIELDS, upsert, manual_features

st.set_page_config(page_title='MCI Dementia Risk', page_icon='🧠', layout='wide')
R = ROOT / 'runs'
BLUE, GREY, RED, GREEN, AMBER = '#1f5fbf', '#9aa5b1', '#c0392b', '#2e8b57', '#d68910'
MODEL_INFO = {
    'M0': ('Clinical', 'Age, sex, education, MMSE, FAQ, ADAS-Cog13, delayed word recall and time since MCI diagnosis'),
    'M1': ('Clinical + MRI', 'Clinical model plus hippocampal volume and entorhinal thickness from the latest MRI'),
    'M2': ('Clinical + MRI + cognitive trend', 'Adds the yearly change in ADAS-Cog13 and FAQ scores'),
    'M3': ('Clinical + MRI + cognitive and atrophy trends', 'Adds the yearly change in hippocampal volume and entorhinal thickness from repeat MRI'),
}
SHORT = {'M0': 'Clinical', 'M1': '+ MRI', 'M2': '+ cognitive trend', 'M3': '+ atrophy trend'}
HORIZON_LABEL = {365: '1 year', 730: '2 years', 1095: '3 years'}
BANDS = [(0.15, 'Lower risk', GREEN), (0.30, 'Intermediate risk', AMBER), (1.01, 'Higher risk', RED)]
FRIENDLY = {**LABELS, 'tau': 'Time since MCI diagnosis', 'sex': 'Sex'}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.card {background:#f3f6fb; border-radius:14px; padding:16px 20px; margin-bottom:10px;}
.card h4 {margin:0 0 4px 0; font-size:0.95rem; color:#5b6776; font-weight:600;}
.card .big {font-size:2.2rem; font-weight:700; line-height:1.1;}
.chip {display:inline-block; padding:3px 12px; border-radius:999px; color:white; font-weight:600; font-size:0.85rem;}
.muted {color:#5b6776; font-size:0.88rem;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- data
def jload(name):
    p = R / name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}


def csv(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def resources():
    return dict(prep=jload('preprocessing.json'), available=jload('secondary_models.json'), paired=jload('models.json'),
                boot_available=jload('bootstrap_secondary.json'), boot_paired=jload('bootstrap_paired.json'))


RES = resources()
prep = RES['prep']


@st.cache_data
def cohort_inputs():
    return pd.read_csv(R / 'train_inputs.csv', parse_dates=['t0'])


@st.cache_data
def cohort_landmarks():
    return pd.read_csv(R / 'train_landmarks.csv', parse_dates=['t0'])


inputs = cohort_inputs()


def risk_curve(model, row, days):
    """Risk of a dementia diagnosis by each day in `days` for one patient."""
    frame = pd.DataFrame([{c: row[c] for c in model['columns']}])
    u = float(np.exp(linear_predictor(model, frame))[0])
    H0 = _exposure(np.asarray(days, float)) @ np.exp(model['log_hazard'])
    return 1 - np.exp(-H0 * u)


def complete(row, model_name):
    return all(row.get(c) is not None and not pd.isna(row.get(c)) for c in MODELS[model_name])


def mri_trend_ok(row):
    s = row.get('mri_span_days')
    return s is not None and not pd.isna(s) and s >= MIN_SPAN


def applicable(row):
    out = [m for m in MODELS if complete(row, m)]
    if 'M3' in out and not mri_trend_ok(row):
        out.remove('M3')
    return out


def band(p):
    for cut, label, color in BANDS:
        if p < cut:
            return label, color


def contributions(model, row):
    cols = model['columns']
    z = (np.array([row[c] for c in cols], float) - np.array(model['mean'])) / np.array(model['scale'])
    c = pd.Series(z * np.array(model['beta']), index=cols)
    if 'tau' in c and 'tau2' in c:
        c['tau'] += c.pop('tau2')
    return c.drop(labels=['field3t'], errors='ignore')


@st.cache_data
def cohort_risk(model_name, set_key, h):
    m = RES[set_key][model_name]
    d = inputs.sort_values('t0').drop_duplicates('RID', keep='last').dropna(subset=MODELS[model_name])
    u = np.exp(linear_predictor(m, d))
    H0 = (_exposure(np.array([h], float)) @ np.exp(m['log_hazard']))[0]
    return 1 - np.exp(-H0 * u)


def sg(v):
    v = round(float(v), 3)
    return f'{v + 0.0:+.3f}' if v != 0 else '0.000'


def fmt_date(v):
    try:
        return str(pd.Timestamp(v).date())
    except Exception:
        return '—'


# --------------------------------------------------------------------------- storage
db = sqlite3.connect(ROOT / 'profiles.sqlite', check_same_thread=False)
db.execute('CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0)')
initialize(db, inputs)


def profiles():
    return pd.read_sql('SELECT * FROM profiles ORDER BY star DESC, name, day', db)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('## 🧠 MCI Dementia Risk')
    st.caption('Estimates the risk that a patient with mild cognitive impairment (MCI) will receive a dementia '
               'diagnosis within 1–3 years.')
    page = st.radio('Navigate', ['Patient risk', 'Add patient', 'Patient list', 'Model performance', 'Study cohort'],
                    label_visibility='collapsed')
    with st.expander('Intended use'):
        st.markdown(
            '- Adults with a confirmed MCI diagnosis and no prior dementia diagnosis.\n'
            '- Not for patients receiving anti-amyloid antibodies (lecanemab, donanemab): treatment changes brain volume.\n'
            '- Supports follow-up planning. It does not diagnose and does not replace clinical judgement.\n'
            f'- Atrophy trends need ≥ 2 MRI scans on the same FreeSurfer version and field strength, ≥ {MIN_SPAN} days apart.\n'
            '- Developed and checked on the ADNI research cohort; performance in other populations is not yet known.')


# =========================================================================== PATIENT RISK
if page == 'Patient risk':
    st.title('Patient risk assessment')
    c1, c2, c3 = st.columns([1.1, 1.6, 1.1])
    source = c1.radio('Patients from', ['My patients', 'ADNI study cohort'], horizontal=True)
    if source == 'My patients':
        pf = profiles()
        pf = pf[~pf.id.str.startswith('Train:')]
        if pf.empty:
            st.info('No patients saved yet. Use **Add patient** or import an ADNI export in **Patient list**.')
            st.stop()
        ix = c2.selectbox('Patient', pf.index, format_func=lambda i: f"{pf.loc[i, 'name']} · visit {pf.loc[i, 'day']}")
        row, visit, name = json.loads(pf.loc[ix, 'features']), pf.loc[ix, 'day'], pf.loc[ix, 'name']
        visits = None
    else:
        rid = c2.selectbox('Participant (RID)', sorted(inputs.RID.unique()), index=0)
        pts = inputs[inputs.RID == rid].sort_values('t0')
        dates = pts.t0.dt.date.astype(str).tolist()
        visit = c3.selectbox('MCI visit', dates, index=len(dates) - 1, help='Defaults to the most recent MCI visit.')
        row = pts[pts.t0.dt.date.astype(str) == visit].iloc[0].to_dict()
        name, visits = f'RID {rid}', pts

    treated = st.toggle('Patient is receiving anti-amyloid antibody therapy')
    if treated:
        st.error('Risk estimates are not available for patients on anti-amyloid antibodies: these drugs cause brain volume '
                 'loss that is unrelated to disease progression, so MRI-based estimates would be misleading.')
        st.stop()

    usable = applicable(row)
    if not usable:
        st.warning('This visit lacks the core clinical scores (MMSE, FAQ, ADAS-Cog13, delayed recall, age, sex, education). '
                   'Add them to obtain an estimate.')
        st.stop()
    b1, b2 = st.columns([2, 1])
    model_name = b1.selectbox('Model', usable[::-1], format_func=lambda m: f'{MODEL_INFO[m][0]}' + (' — most complete for this patient' if m == usable[-1] else ''))
    h = b2.radio('Time frame', [365, 730, 1095], index=1, format_func=lambda d: HORIZON_LABEL[d], horizontal=True)
    if 'M3' not in usable and complete(row, 'M2') and not mri_trend_ok(row):
        st.caption(f'ℹ️ Atrophy-trend model not shown: the MRI scans span less than {MIN_SPAN} days or are missing.')

    model = RES['available'][model_name]
    days = np.arange(30, 1096, 15)
    curve = risk_curve(model, row, days)
    p = float(risk_curve(model, row, [h])[0])
    draws = RES['boot_available'].get(model_name, [])
    lo = hi = None
    if draws:
        sims = np.array([risk_curve(d, row, days) for d in draws])
        band_lo, band_hi = np.quantile(sims, [.025, .975], axis=0)
        ph = np.array([risk_curve(d, row, [h])[0] for d in draws])
        lo, hi = np.quantile(ph, [.025, .975])
    label, color = band(p)

    left, mid, right = st.columns([1.05, 1.5, 1.2])
    with left:
        st.markdown(f"""<div class="card"><h4>Risk of dementia diagnosis within {HORIZON_LABEL[h]}</h4>
        <div class="big" style="color:{color}">{p:.0%}</div>
        <div class="muted">{'95% range ' + f'{lo:.0%}–{hi:.0%}' if lo is not None else ''}</div>
        <div style="margin-top:8px"><span class="chip" style="background:{color}">{label}</span></div></div>""",
                    unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(mode='gauge', value=p * 100, number={'suffix': '%'},
                                       gauge={'axis': {'range': [0, 100], 'ticksuffix': '%'}, 'bar': {'color': color, 'thickness': 0.3},
                                              'steps': [{'range': [0, 15], 'color': '#e3f1e8'}, {'range': [15, 30], 'color': '#fbefd9'},
                                                        {'range': [30, 100], 'color': '#f6dddb'}]}))
        gauge.update_layout(height=190, margin=dict(l=40, r=40, t=20, b=0))
        st.plotly_chart(gauge, width='stretch', config={'displayModeBar': False})
        st.caption('Bands: lower < 15%, intermediate 15–30%, higher ≥ 30% — the range of decision thresholds used to evaluate '
                   'the model; not a clinical guideline.')
    with mid:
        fig = go.Figure()
        if draws:
            fig.add_trace(go.Scatter(x=np.r_[days, days[::-1]] / 30.44, y=np.r_[band_hi, band_lo[::-1]] * 100, fill='toself',
                                     fillcolor='rgba(31,95,191,0.15)', line=dict(width=0), hoverinfo='skip', name='95% range'))
        fig.add_trace(go.Scatter(x=days / 30.44, y=curve * 100, line=dict(color=BLUE, width=3), name='Estimated risk',
                                 hovertemplate='%{x:.0f} months: %{y:.1f}%<extra></extra>'))
        fig.add_vline(x=h / 30.44, line_dash='dot', line_color=GREY)
        fig.update_layout(title='Risk over the next 3 years', height=330, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title='Months from this visit', yaxis_title='Cumulative risk (%)', showlegend=False,
                          yaxis=dict(rangemode='tozero'))
        st.plotly_chart(fig, width='stretch')
    with right:
        cr = cohort_risk(model_name, 'available', h)
        pct = float((cr < p).mean() * 100)
        fig = px.histogram(x=cr * 100, nbins=30, color_discrete_sequence=[GREY])
        fig.add_vline(x=p * 100, line_color=color, line_width=3)
        fig.update_layout(title=f'Higher than {pct:.0f}% of study patients', height=330, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title=f'Risk within {HORIZON_LABEL[h]} (%)', yaxis_title='Patients', bargap=0.05)
        st.plotly_chart(fig, width='stretch')

    d1, d2 = st.columns([1.3, 1])
    with d1:
        c = contributions(model, row)
        c = c.reindex(c.abs().sort_values(ascending=False).index[:8]).sort_values()
        names = [FRIENDLY.get(k, k).split(' (')[0] for k in c.index]
        fig = go.Figure(go.Bar(x=c.values, y=names, orientation='h',
                               marker_color=[RED if v > 0 else GREEN for v in c.values],
                               hovertemplate='%{y}: %{x:+.2f}<extra></extra>'))
        fig.update_layout(title='What drives this estimate', height=360, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_title='← lowers risk · compared with an average study patient · raises risk →')
        st.plotly_chart(fig, width='stretch')
        st.caption('Top 8 factors. Bar length is the effect on the log hazard relative to an average study patient; '
                   'the model also adjusts for MRI field strength.')
    with d2:
        st.markdown('#### Patient at this visit')
        sex = {1.0: 'Male', 2.0: 'Female'}.get(row.get('sex'), '—')
        tiles = [('Age', row.get('age'), '{:.0f}'), ('Education (y)', row.get('education'), '{:.0f}'),
                 ('MMSE /30', row.get('mmse'), '{:.0f}'), ('FAQ /30', row.get('faq'), '{:.0f}'),
                 ('ADAS-Cog13 /85', row.get('adas13'), '{:.1f}'), ('Delayed recall /15', row.get('ravlt_delay'), '{:.0f}'),
                 ('Hippocampus (mm³)', row.get('hippo_adj'), '{:,.0f}'), ('Entorhinal (mm)', row.get('entor'), '{:.2f}'),
                 ('Hippocampus Δ (mm³/yr)', row.get('hippo_slope'), '{:+.0f}')]
        cols = st.columns(3)
        for i, (lab, v, f) in enumerate(tiles):
            cols[i % 3].metric(lab, f.format(v) if v is not None and not pd.isna(v) else '—')
        st.caption(f"{name} · Sex: {sex} · Visit {visit}")
        with st.expander('Where these values come from'):
            st.markdown(
                f"- Latest MRI: **{fmt_date(row.get('mri_date'))}**"
                + (f" · scanner {row.get('scanner')}" if row.get('scanner') not in (None, 'NA') and not pd.isna(row.get('scanner')) else '')
                + f"\n- MRI trend uses scans from **{fmt_date(row.get('mri_from'))}** ({row.get('mri_span_days') or '—'} days)"
                + f"\n- ADAS-Cog13 measured **{fmt_date(row.get('adas13_date'))}**, trend from **{fmt_date(row.get('adas13_from'))}**"
                + f"\n- Model: **{MODEL_INFO[model_name][0]}** — {MODEL_INFO[model_name][1]}. "
                + f"Trained on {model.get('persons_train', '—')} ADNI participants.")
    if visits is not None and len(visits) > 1:
        traj = []
        for _, v in visits.iterrows():
            r2 = v.to_dict()
            if complete(r2, 'M0'):
                traj.append(dict(visit=v.t0, risk=float(risk_curve(RES['available']['M0'], r2, [h])[0]) * 100))
        if len(traj) > 1:
            t = pd.DataFrame(traj)
            fig = px.line(t, x='visit', y='risk', markers=True, color_discrete_sequence=[BLUE])
            fig.update_layout(title=f'Clinical-model risk at each MCI visit ({HORIZON_LABEL[h]} time frame)', height=280,
                              margin=dict(l=10, r=10, t=40, b=10), yaxis_title='Risk (%)', xaxis_title='')
            st.plotly_chart(fig, width='stretch')

    with st.expander('Compare all four models for this patient'):
        if not complete(row, 'M3') or not mri_trend_ok(row):
            st.info('Needs all inputs including an MRI atrophy trend.')
        else:
            out = []
            for m in MODELS:
                pm = RES['paired'][m]
                val = float(risk_curve(pm, row, [h])[0])
                q = np.quantile([risk_curve(d, row, [h])[0] for d in RES['boot_paired'][m]], [.025, .975])
                out.append(dict(model=MODEL_INFO[m][0], risk=val * 100, lo=q[0] * 100, hi=q[1] * 100))
            o = pd.DataFrame(out)
            fig = go.Figure(go.Bar(x=o.risk, y=o.model, orientation='h', marker_color=[GREY, GREY, GREY, BLUE],
                                   error_x=dict(type='data', symmetric=False, array=o.hi - o.risk, arrayminus=o.risk - o.lo)))
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=f'Risk within {HORIZON_LABEL[h]} (%)')
            st.plotly_chart(fig, width='stretch')
            st.caption('All four models here are trained on the same patients so they can be compared fairly. Differences '
                       'for a single patient are not evidence that one model is better — see Model performance.')

# =========================================================================== ADD PATIENT
elif page == 'Add patient':
    st.title('Add a patient')
    st.caption('Enter the values from the current MCI visit. Leave MRI and history tables empty if not available; '
               'the app then uses the most complete model the data allow.')
    with st.form('new_patient', clear_on_submit=False):
        st.markdown('#### Patient')
        a, b, c, d = st.columns(4)
        name = a.text_input('Patient name or ID')
        visit = b.date_input('Visit date', value=pd.Timestamp.today().date(), min_value=pd.Timestamp('1990-01-01').date())
        mci_since = c.number_input('Years since MCI was first diagnosed', 0.0, 20.0, 1.0, step=0.5)
        sex = d.radio('Sex', ['Male', 'Female'], horizontal=True)
        a, b = st.columns(2)
        age = a.slider('Age (years)', 50, 100, 73)
        edu = b.slider('Education (years)', 0, 25, 16)

        st.markdown('#### Cognition and daily function')
        a, b, c, d = st.columns(4)
        mmse = a.slider('MMSE (0–30)', 0, 30, 27, help='Higher is better.')
        faq = b.slider('FAQ (0–30)', 0, 30, 3, help='Functional Activities Questionnaire. Higher means more difficulty.')
        adas = c.slider('ADAS-Cog13 (0–85)', 0.0, 85.0, 17.0, step=0.5, help='Higher means more impairment.')
        ravlt = d.slider('Delayed word recall, RAVLT (0–15)', 0, 15, 4)

        st.markdown('#### MRI at this visit (FreeSurfer)')
        a, b, c, d, e = st.columns(5)
        mri_date = a.date_input('MRI date', value=None, min_value=pd.Timestamp('1990-01-01').date())
        hv = b.number_input('Hippocampus L+R (mm³)', 2000.0, 14000.0, value=None, step=50.0)
        icv = c.number_input('Intracranial vol. (mm³)', 900000.0, 2200000.0, value=None, step=10000.0)
        ent = d.number_input('Entorhinal mean (mm)', 1.0, 6.0, value=None, step=0.05)
        field = e.radio('Field strength', ['3T', '1.5T'], horizontal=True)

        st.markdown('#### Earlier measurements (optional, for trends)')
        st.caption(f'Trends need at least one earlier value ≥ {MIN_SPAN} days before this visit. MRI scans must use the same '
                   'FreeSurfer version and field strength as the current scan.')
        st.markdown('**Earlier cognitive tests**')
        cog_hist = st.data_editor(pd.DataFrame({'Date': pd.Series(dtype='datetime64[ns]'), 'ADAS-Cog13': pd.Series(dtype=float),
                                               'FAQ': pd.Series(dtype=float)}), num_rows='dynamic', key='cog_hist',
                                 column_config={'Date': st.column_config.DateColumn('Date')}, width='stretch')
        st.markdown('**Earlier MRI scans**')
        mri_hist = st.data_editor(pd.DataFrame({'Date': pd.Series(dtype='datetime64[ns]'), 'Hippocampal volume (mm³)': pd.Series(dtype=float),
                                               'Intracranial volume (mm³)': pd.Series(dtype=float),
                                               'Entorhinal thickness (mm)': pd.Series(dtype=float)}),
                                 num_rows='dynamic', key='mri_hist', column_config={'Date': st.column_config.DateColumn('Date')},
                                 width='stretch')
        confirm = st.checkbox('MCI confirmed at this visit, no previous dementia diagnosis, not on anti-amyloid therapy')
        saved = st.form_submit_button('Save patient', type='primary')

    if saved:
        if not name.strip() or not confirm:
            st.error('Enter a patient name or ID and confirm the eligibility statement.')
        else:
            v = pd.Timestamp(visit)
            f = manual_features(prep, visit=v, mci_years=mci_since, age=age, sex=sex, education=edu, mmse=mmse, faq=faq,
                                adas13=adas, ravlt=ravlt, mri_date=mri_date, hv=hv, icv=icv, entor=ent, field=field,
                                cog_history=cog_hist, mri_history=mri_hist)
            mh = mri_hist.dropna(subset=['Date'])
            with db:
                upsert(db, [(f'Manual:{uuid.uuid4()}', name.strip(), str(v.date()), json.dumps(f))])
            avail = applicable(f)
            st.success(f"Saved **{name.strip()}**. Available model: **{MODEL_INFO[avail[-1]][0]}**. "
                       'Open **Patient risk** to see the estimate.')
            if 'hippo_slope' in f:
                st.caption(f"Hippocampal change {f['hippo_slope']:+.0f} mm³/year over {f['mri_span_days']} days.")
            elif 'hippo_adj' in f and len(mh):
                st.caption(f'MRI history spans {f.get("mri_span_days", 0)} days (< {MIN_SPAN}); atrophy trend not used.')

# =========================================================================== PATIENT LIST
elif page == 'Patient list':
    st.title('Patient list')
    pf = profiles()
    mine = pf[~pf.id.str.startswith('Train:')].copy()
    with st.expander('Import patients from an ADNI data export (ZIP)'):
        st.caption('The ZIP must contain DXSUM, MMSE, FAQ, ADAS, NEUROBAT, PTDEMOG and UCSFFSX7 CSV files '
                   '(FOXLABBSI and All_Subjects_Structural_MRI_Images are optional). Each participant is imported at their most '
                   'recent MCI visit; importing again adds new visits and updates existing ones.')
        up = st.file_uploader('ADNI export', type='zip', label_visibility='collapsed')
        if up and st.button('Import'):
            try:
                with st.spinner('Reading files and preparing patients…'):
                    r = import_zip(db, up, prep, set(inputs.RID))
                st.success(f"Added {r['added']} and updated {r['updated']} patient visits from {r['rids']} participants. "
                           f"{r['skipped_train']} participants were skipped because they are in the model's training data.")
                st.rerun()
            except Exception as exc:
                st.error(f'Import failed: {exc}')
    if mine.empty:
        st.info('No patients yet.')
        st.stop()
    feats = mine.features.map(json.loads)
    view = pd.DataFrame({
        '★': mine.star.map({1: '★', 0: ''}).values, 'Patient': mine.name.values, 'Visit': mine.day.values,
        'Age': feats.map(lambda f: f.get('age')).values, 'MMSE': feats.map(lambda f: f.get('mmse')).values,
        'ADAS-Cog13': feats.map(lambda f: f.get('adas13')).values, 'Latest MRI': feats.map(lambda f: f.get('mri_date')).values,
        'Best model': feats.map(lambda f: MODEL_INFO[applicable(f)[-1]][0] if applicable(f) else 'Incomplete').values})
    q = st.text_input('Search by name or ID', placeholder='Type to filter…')
    keep = view.Patient.str.contains(q, case=False, regex=False)
    k1, k2, k3 = st.columns(3)
    k1.metric('Patients', int(keep.sum()))
    k2.metric('With MRI', int(view[keep]['Latest MRI'].notna().sum()))
    k3.metric('With atrophy trend', int((view[keep]['Best model'] == MODEL_INFO['M3'][0]).sum()))
    st.dataframe(view[keep].assign(**{'Latest MRI': view[keep]['Latest MRI'].fillna('—')}), hide_index=True, width='stretch',
                 column_config={'Age': st.column_config.NumberColumn(format='%d'),
                                'ADAS-Cog13': st.column_config.NumberColumn(format='%.1f')})
    sel = st.selectbox('Select a patient', mine.index[keep.values], format_func=lambda i: f"{mine.loc[i, 'name']} · {mine.loc[i, 'day']}")
    a, b, c = st.columns([1, 1, 2])
    if a.button('☆ Unstar' if mine.loc[sel, 'star'] else '★ Star'):
        with db:
            db.execute('UPDATE profiles SET star=1-star WHERE id=?', (mine.loc[sel, 'id'],))
        st.rerun()
    sure = c.checkbox('Confirm deletion')
    if b.button('Delete', disabled=not sure):
        with db:
            db.execute('DELETE FROM profiles WHERE id=?', (mine.loc[sel, 'id'],))
        st.rerun()

# =========================================================================== MODEL PERFORMANCE
elif page == 'Model performance':
    st.title('How well do the risk estimates work?')
    st.caption('Results from repeated cross-validation on the training data (5 folds × 20 repeats, one visit per patient) and '
               'from a separate held-out test set. AUC: how well the model ranks patients who are later diagnosed above those '
               'who are not (0.5 = chance, 1 = perfect).')
    sec = {m: csv(f'cv/SEC_{m}_metrics.csv') for m in MODELS}
    secinfo = {m: jload(f'cv/SEC_{m}_info.json') for m in MODELS}
    tcov = csv('test_coverage.csv')
    rows = []
    for m in MODELS:
        s = sec[m]
        if s is None:
            continue
        s = s[s.scope == 'random']
        for _, r in s.iterrows():
            t = tcov[(tcov.model == m) & (tcov.h == r.h) & (tcov.scope == 'random')] if tcov is not None else pd.DataFrame()
            rows.append(dict(model=MODEL_INFO[m][0], code=m, h=int(r.h), auc_cv=r.auc, auc_test=t.auc.iloc[0] if len(t) else np.nan,
                             oe=r.oe_turnbull, ipa=r.ipa, nb=r.nb_core, persons=secinfo[m].get('persons')))
    perf = pd.DataFrame(rows)
    best = perf[(perf.code == 'M3') & (perf.h == 730)].iloc[0]
    k = st.columns(4)
    k[0].metric('Accuracy at 2 years (AUC)', f'{best.auc_cv:.2f}', help='Most complete model, cross-validation.')
    k[1].metric('Held-out test AUC', f'{best.auc_test:.2f}')
    k[2].metric('Predicted vs observed cases', f'{1 / best.oe:.2f}×', help='Ratio of predicted to observed diagnoses (1.00 = perfect).')
    k[3].metric('Patients used to build models', f"{int(perf.persons.max()):,}")

    t1, t2, t3, t4, t5 = st.tabs(['Accuracy', 'Calibration', 'Clinical usefulness', 'Model profiles', 'Does repeat MRI help?'])
    with t1:
        a, b = st.columns(2)
        perf['short'] = perf.code.map(SHORT)
        fig = px.line(perf, x='h', y='auc_cv', color='short', markers=True,
                      labels={'h': 'Time frame (days)', 'auc_cv': 'AUC', 'model': ''},
                      color_discrete_sequence=[GREY, '#7fa7de', '#4a7fcf', BLUE])
        fig.update_layout(title='Accuracy by time frame (cross-validation)', height=380, legend=dict(orientation='h', y=-0.25),
                          xaxis=dict(tickvals=[365, 730, 1095], ticktext=['1 year', '2 years', '3 years']))
        a.plotly_chart(fig, width='stretch')
        p2 = perf[perf.h == 730].melt(id_vars='short', value_vars=['auc_cv', 'auc_test'], var_name='set', value_name='AUC')
        p2['set'] = p2.set.map({'auc_cv': 'Cross-validation', 'auc_test': 'Held-out test'})
        fig = px.bar(p2, x='short', y='AUC', color='set', barmode='group', color_discrete_sequence=[BLUE, GREY], text_auto='.2f')
        fig.update_layout(title='Cross-validation vs held-out test at 2 years', height=380, yaxis_range=[0.5, 1],
                          legend=dict(orientation='h', y=-0.25, title=''), xaxis_title='')
        b.plotly_chart(fig, width='stretch')
    with t2:
        cal = csv('test_calibration.csv')
        hh = st.radio('Time frame', [365, 730, 1095], index=1, format_func=lambda d: HORIZON_LABEL[d], horizontal=True, key='calh')
        c = cal[(cal.h == hh)].copy()
        c['model'] = c.model.map(SHORT)
        a, b = st.columns([1.3, 1])
        fig = px.line(c.sort_values(['model', 'predicted']), x='predicted', y='observed_turnbull', color='model', markers=True,
                         color_discrete_sequence=[GREY, '#7fa7de', '#4a7fcf', BLUE],
                         labels={'predicted': 'Predicted risk', 'observed_turnbull': 'Observed rate', 'model': ''})
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#777'), name='Perfect agreement'))
        fig.update_layout(title='Predicted vs observed (held-out test, 10 risk groups)', height=420,
                          xaxis=dict(range=[0, 1], tickformat='.0%'), yaxis=dict(range=[0, 1], tickformat='.0%'),
                          legend=dict(orientation='h', y=-0.2))
        a.plotly_chart(fig, width='stretch')
        oe = perf[perf.h == hh][['short', 'oe']].assign(ratio=lambda d: 1 / d.oe)
        fig = px.bar(oe, x='ratio', y='short', orientation='h', text_auto='.2f', color_discrete_sequence=[BLUE])
        fig.add_vline(x=1, line_dash='dash', line_color='#555')
        fig.update_layout(title='Predicted ÷ observed diagnoses (cross-validation)', height=420, xaxis_title='1.00 = perfect',
                          yaxis_title='', xaxis_range=[0.7, 1.3])
        b.plotly_chart(fig, width='stretch')
        st.caption('Observed rates use the Turnbull estimator, which accounts for dementia being recognised only at scheduled visits.')
    with t3:
        mm = csv('cv/main_metrics.csv')
        m = mm[(mm.scope == 'random') & (mm.h == 730)]
        th = [5, 10, 15, 20, 25, 30, 35, 40]
        nb = []
        for _, r in m.iterrows():
            for t in th:
                nb.append(dict(model=SHORT[r.model], threshold=t, nb=r[f'nb_{t}'] * 100))
        obs = float(m.observed_ipcw.iloc[0])
        for t in th:
            nb.append(dict(model='Follow everyone closely', threshold=t, nb=(obs - (1 - obs) * t / (100 - t)) * 100))
            nb.append(dict(model='Follow no one closely', threshold=t, nb=0.0))
        nb = pd.DataFrame(nb)
        fig = px.line(nb, x='threshold', y='nb', color='model', markers=True,
                      color_discrete_sequence=[GREY, '#7fa7de', '#4a7fcf', BLUE, AMBER, '#333'],
                      labels={'threshold': 'Risk threshold for closer follow-up (%)', 'nb': 'Net benefit (per 100 patients)', 'model': ''})
        fig.add_vrect(x0=15, x1=30, fillcolor='#eef3fb', line_width=0, layer='below')
        fig.update_layout(title='Decision curve at 2 years: correctly flagged patients per 100, after penalising false alarms',
                          height=440, legend=dict(orientation='h', y=-0.25), yaxis_range=[-5, max(nb.nb) + 3])
        st.plotly_chart(fig, width='stretch')
        st.caption('A model is useful where its line is above both "follow everyone" and "follow no one". The shaded band '
                   '(15–30%) is the range used to judge the models.')
    with t4:
        p = perf[perf.h == 730].copy()
        metrics = {'Accuracy (AUC)': 'auc_cv', 'Held-out AUC': 'auc_test', 'Explained risk (IPA)': 'ipa',
                   'Net benefit': 'nb', 'Patients covered': 'persons'}
        fig = go.Figure()
        for i, r in p.iterrows():
            vals = []
            for lab, col in metrics.items():
                lo_, hi_ = p[col].min(), p[col].max()
                vals.append(0.3 + 0.7 * (r[col] - lo_) / (hi_ - lo_) if hi_ > lo_ else 1)
            fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=list(metrics) + list(metrics)[:1], fill='toself',
                                          name=SHORT[r.code], opacity=0.5,
                                          line=dict(color=[GREY, '#7fa7de', '#4a7fcf', BLUE][list(MODELS).index(r.code)])))
        fig.update_layout(title='Trade-offs between models at 2 years (scaled: outer edge = best of the four)', height=480,
                          polar=dict(radialaxis=dict(visible=False, range=[0, 1])), legend=dict(orientation='h', y=-0.1))
        a, b = st.columns([1.3, 1])
        a.plotly_chart(fig, width='stretch')
        tab = p[['short', 'auc_cv', 'auc_test', 'ipa', 'nb', 'persons']].rename(columns={
            'short': 'Model', 'auc_cv': 'AUC (CV)', 'auc_test': 'AUC (test)', 'ipa': 'IPA', 'nb': 'Net benefit', 'persons': 'Patients'})
        b.dataframe(tab.round(3), hide_index=True, width='stretch')
        b.caption('Richer models are slightly more accurate but need more data, so they apply to fewer patients.')
    with t5:
        summ = jload('summary.json')
        if summ:
            st.info(f"**Conclusion:** {summ['overall']}. {summ['test_note']}")
            rows = []
            for hz, t in summ['tiers'].items():
                for key, lab in [('cv', 'Cross-validation'), ('test', 'Held-out test')]:
                    x = t[key]
                    rows.append({'Time frame': HORIZON_LABEL[int(hz)], 'Data': lab,
                                 'AUC gain from atrophy trend': f"{sg(x['d_auc'])} ({sg(x['auc_ci'][0])} to {sg(x['auc_ci'][1])})",
                                 'Net-benefit gain': f"{sg(x['d_nb'])} ({sg(x['nb_ci'][0])} to {sg(x['nb_ci'][1])})",
                                 'Verdict': x['conclusion']})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
            st.caption(f"Verdict rule set before analysis: 'Added value' needs an AUC gain ≥ {summ['protocol']['delta_auc']} with a "
                       f"positive lower bound and a positive net-benefit gain; 'Equivalent' needs the AUC interval within "
                       f"±{summ['protocol']['delta_auc']} and the net-benefit gain ≤ {summ['protocol']['nb_margin']}.")
        s = csv('sensitivity_summary.csv')
        if s is not None:
            s = s[s.comparison.isin(['M3 − M2', 'M3_twopoint − M2', 'M2_bsi − M2'])].copy()
            tag = s.analysis.str.extract(r'^(S\d+[a-z]?|L1)')[0].fillna('main')
            s = s[~(tag.isin(['S1', 'S2a']) & (s.comparison == 'M3 − M2'))]
            tag = tag[s.index]
            s['label'] = s.analysis.str.replace(r'^(S\d+[a-z]?|L1) · ', '', regex=True)
            s.loc[s.comparison == 'M3_twopoint − M2', 'label'] = 'Atrophy rate from only two scans'
            s.loc[(tag == 'S2b') & (s.comparison == 'M3 − M2'), 'label'] = 'ADNI-GO/2 subgroup'
            s.loc[s.comparison == 'M2_bsi − M2', 'label'] = 'ADNI-GO/2: boundary-shift integral instead'
            s['label'] = s.label.str.slice(0, 1).str.upper() + s.label.str.slice(1)
            s = s.iloc[::-1]
            fig = go.Figure(go.Scatter(x=s.d_auc, y=s.label, mode='markers', marker=dict(color=BLUE, size=9),
                                       error_x=dict(type='data', symmetric=False, array=s.auc_hi - s.d_auc, arrayminus=s.d_auc - s.auc_lo)))
            fig.add_vrect(x0=-0.02, x1=0.02, fillcolor='#eef3fb', line_width=0, layer='below')
            fig.add_vline(x=0, line_color='#555')
            fig.update_layout(title='AUC gain from MRI atrophy trend at 2 years, across alternative analyses', height=560,
                              xaxis_title='AUC gain (95% interval) · shaded = ±0.02 equivalence margin', margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, width='stretch')
        mm = csv('cv/main_metrics.csv')
        if mm is not None:
            m3 = mm[(mm.scope == 'random') & (mm.model == 'M3')]
            st_ = m3.melt(id_vars='h', value_vars=['case_certain', 'ambiguous', 'control_certain', 'censored_early'],
                          var_name='status', value_name='patients')
            st_['status'] = st_.status.map({'case_certain': 'Diagnosed before the time point', 'ambiguous': 'Diagnosed at the first visit after it',
                                            'control_certain': 'Still dementia-free after it', 'censored_early': 'Left follow-up before it'})
            st_['Time frame'] = st_.h.map(HORIZON_LABEL)
            fig = px.bar(st_, x='patients', y='Time frame', color='status', orientation='h',
                         color_discrete_sequence=[RED, '#e8a0a0', '#cfd8e3', '#7a869a'])
            fig.update_layout(title='What we know about each patient at each time point', height=300, barmode='stack',
                              legend=dict(orientation='h', y=-0.35, title=''), xaxis_title='Patients (average per repeat)')
            st.plotly_chart(fig, width='stretch')

# =========================================================================== STUDY COHORT
else:
    st.title('Study cohort')
    st.caption('ADNI participants used to build the models (training portion, 80% of participants).')
    lm = cohort_landmarks()
    first = lm.sort_values('t0').drop_duplicates('RID').copy()
    first['Outcome within 3 years'] = np.where(first.R <= 1095, 'Diagnosed with dementia',
                                               np.where(first.L >= 1095, 'Dementia-free', 'Follow-up < 3 years'))
    first['Sex'] = first.sex.map({1.0: 'Male', 2.0: 'Female'})
    dx = read(ROOT / 'data' / 'train', 'DXSUM').dropna(subset=['EXAMDATE'])
    latest = dx[dx.DIAGNOSIS.isin([1, 2, 3])].sort_values('EXAMDATE').drop_duplicates('RID', keep='last')
    k = st.columns(4)
    k[0].metric('Participants with a diagnosis', f'{latest.RID.nunique():,}')
    k[1].metric('Patients with MCI visits', f'{first.RID.nunique():,}')
    k[2].metric('MCI visits analysed', f'{len(lm):,}')
    k[3].metric('Later diagnosed with dementia', f"{int(lm.groupby('RID').event.max().sum()):,}")
    OUTC = {'Diagnosed with dementia': RED, 'Dementia-free': GREEN, 'Follow-up < 3 years': GREY}

    a, b, c = st.columns(3)
    d = latest.DIAGNOSIS.map({1: 'Cognitively normal', 2: 'MCI', 3: 'Dementia'}).value_counts().reset_index()
    fig = px.pie(d, names='DIAGNOSIS', values='count', hole=0.5, color='DIAGNOSIS',
                 color_discrete_map={'Cognitively normal': GREEN, 'MCI': AMBER, 'Dementia': RED})
    fig.update_layout(title='Most recent diagnosis', height=340, legend=dict(orientation='h', y=-0.1))
    a.plotly_chart(fig, width='stretch')
    d = first['Outcome within 3 years'].value_counts().reset_index()
    fig = px.pie(d, names='Outcome within 3 years', values='count', hole=0.5, color='Outcome within 3 years', color_discrete_map=OUTC)
    fig.update_layout(title='Outcome within 3 years of the first MCI visit', height=340, legend=dict(orientation='h', y=-0.1))
    b.plotly_chart(fig, width='stretch')
    d = first.Sex.value_counts().reset_index()
    fig = px.pie(d, names='Sex', values='count', hole=0.5, color_discrete_sequence=[BLUE, '#9bb8e6'])
    fig.update_layout(title='Sex', height=340, legend=dict(orientation='h', y=-0.1))
    c.plotly_chart(fig, width='stretch')

    a, b = st.columns(2)
    fig = px.histogram(first, x='age', color='Outcome within 3 years', nbins=30, barmode='stack',
                       color_discrete_map=OUTC, labels={'age': 'Age at first MCI visit (years)'})
    fig.update_layout(title='Age distribution', height=360, legend=dict(orientation='h', y=-0.3, title=''), yaxis_title='Patients')
    a.plotly_chart(fig, width='stretch')
    fig = px.histogram(first, x='mmse', color='Outcome within 3 years', barmode='group', color_discrete_map=OUTC,
                       labels={'mmse': 'MMSE at first MCI visit'})
    fig.update_layout(title='MMSE distribution', height=360, legend=dict(orientation='h', y=-0.3, title=''), yaxis_title='Patients')
    b.plotly_chart(fig, width='stretch')

    a, b = st.columns(2)
    sub = first.dropna(subset=['hippo_adj'])
    fig = px.violin(sub, x='Outcome within 3 years', y='hippo_adj', color='Outcome within 3 years', box=True, points=False,
                    color_discrete_map=OUTC, labels={'hippo_adj': 'Hippocampal volume, head-size adjusted (mm³)'})
    fig.update_layout(title='Hippocampal volume by later outcome', height=400, showlegend=False, xaxis_title='',
                      yaxis_range=[3000, 10500])
    a.plotly_chart(fig, width='stretch')
    sub = first.dropna(subset=['hippo_adj', 'adas13'])
    fig = px.scatter(sub, x='adas13', y='hippo_adj', color='Outcome within 3 years', opacity=0.6, color_discrete_map=OUTC,
                     labels={'adas13': 'ADAS-Cog13', 'hippo_adj': 'Hippocampal volume (mm³)'})
    fig.update_layout(title='Cognitive testing and hippocampal volume', height=400, legend=dict(orientation='h', y=-0.25, title=''),
                      yaxis_range=[3000, 10500])
    b.plotly_chart(fig, width='stretch')

    a, b = st.columns([1.5, 1])
    ev = lm[np.isfinite(lm.R) & (lm.R <= 1825)]
    fig = px.histogram(ev, x=ev.R / 30.44, nbins=60, color_discrete_sequence=[BLUE])
    fig.update_layout(title='When dementia was recorded after an MCI visit', height=360, bargap=0.05,
                      xaxis_title='Months after the MCI visit', yaxis_title='Diagnoses')
    a.plotly_chart(fig, width='stretch')
    a.caption('Peaks at 6, 12, 18 and 24 months follow the study visit schedule: dementia is only recognised at a visit. '
              'The models account for this.')
    conf = csv('conflicts.csv')
    if conf is not None:
        kind = conf.action.map(lambda a: 'Repeat scans averaged' if a == 'averaged' else
                               'Same-day 1.5T/3T scans: one kept' if a.startswith('kept') else
                               'Inconsistent demographics' if a.startswith('latest record') else
                               'Conflicting test score removed' if a == 'measurement removed' else 'Conflicting diagnosis removed')
        cc = kind.value_counts().rename_axis('action').reset_index(name='records')
        fig = px.pie(cc, names='action', values='records', hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(title='Data quality checks applied', height=360, legend=dict(orientation='h', y=-0.2))
        b.plotly_chart(fig, width='stretch')
        b.download_button('Download data-quality log (CSV)', conf.to_csv(index=False).encode('utf-8'), 'data_quality_log.csv')
