"""Shared look, cached loaders and small helpers for the app pages."""
import numpy as np
import pandas as pd
import streamlit as st

from ..config import DB_PATH, HORIZONS, LABELS, MODELS, PRIMARY, RUNS_DIR
from ..files import load_json
from ..model import predict
from ..risk import BAND_LOWER, BAND_UPPER
from ..store import PatientStore

BLUE, GREY, RED, GREEN, AMBER = '#1f5fbf', '#9aa5b1', '#c0392b', '#2e8b57', '#d68910'
MODEL_COLORS = [GREY, '#7fa7de', '#4a7fcf', BLUE]               # M0 ... M3
BAND_COLORS = {'low': GREEN, 'mid': AMBER, 'high': RED}
BAND_FILL = {'low': '#e3f1e8', 'mid': '#fbefd9', 'high': '#f6dddb'}
MODEL_INFO = {
    'M0': ('Clinical', 'Age, sex, education, MMSE, FAQ, ADAS-Cog13, delayed word recall and time since MCI diagnosis'),
    'M1': ('Clinical + MRI', 'Clinical model plus hippocampal volume and entorhinal thickness from the latest MRI'),
    'M2': ('Clinical + MRI + cognitive trend', 'Adds the yearly change in ADAS-Cog13 and FAQ scores'),
    'M3': ('Clinical + MRI + cognitive and atrophy trends',
           'Adds the yearly change in hippocampal volume and entorhinal thickness from repeat MRI'),
}
SHORT = {'M0': 'Clinical', 'M1': '+ MRI', 'M2': '+ cognitive trend', 'M3': '+ atrophy trend'}
HORIZON_LABEL = {h: f'{round(h / 365)} year{"s" if round(h / 365) > 1 else ""}' for h in HORIZONS}
FRIENDLY = {**LABELS, 'tau': 'Time since MCI diagnosis'}
BAND_TEXT = (f'Bands: lower < {BAND_LOWER:.0%}, intermediate {BAND_LOWER:.0%}–{BAND_UPPER:.0%}, '
             f'higher ≥ {BAND_UPPER:.0%}')
PRODUCED_BY = [  # (file name prefix, script that writes it)
    ('cv/', 'validation.py'), ('bootstrap_', 'uncertainty.py'), ('test_', 'evaluate.py'),
    ('summary', 'report.py'), ('sensitivity_summary', 'report.py'), ('', 'train.py'),
]
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
.card {background:#f3f6fb; border-radius:14px; padding:16px 20px; margin-bottom:10px;}
.card h4 {margin:0 0 4px 0; font-size:0.95rem; color:#5b6776; font-weight:600;}
.card .big {font-size:2.2rem; font-weight:700; line-height:1.1;}
.chip {display:inline-block; padding:3px 12px; border-radius:999px; color:white; font-weight:600; font-size:0.85rem;}
.muted {color:#5b6776; font-size:0.88rem;}
</style>"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def plot(fig, where=st):
    where.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------- results on disk
def missing(*names):
    return [n for n in names if not (RUNS_DIR / n).exists()]


def require(*names):
    """True when every results file exists; otherwise show which scripts produce the missing ones."""
    absent = missing(*names)
    if absent:
        scripts = sorted({next(s for prefix, s in PRODUCED_BY if n.startswith(prefix)) for n in absent})
        st.warning(f'Results not available yet: {", ".join(absent)}. Run {", ".join(scripts)} first.')
    return not absent


@st.cache_data
def run_json(name):
    path = RUNS_DIR / name
    return load_json(path) if path.exists() else {}


@st.cache_data
def run_csv(name):
    path = RUNS_DIR / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def resources():
    return dict(prep=run_json('preprocessing.json'), available=run_json('secondary_models.json'),
                paired=run_json('models.json'), boot_available=run_json('bootstrap_secondary.json'),
                boot_paired=run_json('bootstrap_paired.json'))


@st.cache_resource
def patient_store():
    return PatientStore(DB_PATH)


@st.cache_data
def cohort_inputs():
    return pd.read_csv(RUNS_DIR / 'train_inputs.csv', parse_dates=['t0'])


@st.cache_data
def cohort_landmarks():
    return pd.read_csv(RUNS_DIR / 'train_landmarks.csv', parse_dates=['t0'])


@st.cache_data
def cohort_risk(model_name, set_key, h):
    """Risk at h for each training participant's latest visit (for comparing a patient with the cohort)."""
    latest = cohort_inputs().sort_values('t0').drop_duplicates('RID', keep='last').dropna(subset=MODELS[model_name])
    return predict(resources()[set_key][model_name], latest, h)


# --------------------------------------------------------------------------- formatting
def fmt_signed(v):
    v = round(float(v), 3)
    return f'{v + 0.0:+.3f}' if v != 0 else '0.000'


def fmt_date(v):
    try:
        return str(pd.Timestamp(v).date())
    except (TypeError, ValueError):
        return '—'


def fmt_value(v, pattern):
    return '—' if v is None or (isinstance(v, float) and np.isnan(v)) else pattern.format(v)


def horizon_radio(where=st, key=None):
    return where.radio('Time frame', HORIZONS, index=HORIZONS.index(PRIMARY), format_func=HORIZON_LABEL.get,
                       horizontal=True, key=key)
