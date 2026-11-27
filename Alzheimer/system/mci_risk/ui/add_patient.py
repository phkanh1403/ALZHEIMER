"""Page: enter one visit by hand, with optional earlier tests and scans for trends."""
import json
import uuid

import pandas as pd
import streamlit as st

from ..config import MIN_SPAN, MRI_MAX_AGE
from ..manual_entry import (
    DATE,
    FIELD_STRENGTHS,
    MRI_HISTORY,
    InvalidEntry,
    empty_cognitive_history,
    empty_mri_history,
    manual_features,
)
from ..risk import applicable_models
from ..store import ProfileRow
from . import common as ui

EARLIEST_DATE = pd.Timestamp('1990-01-01').date()


def _history_editors():
    st.markdown('**Earlier cognitive tests**')
    cog = st.data_editor(empty_cognitive_history(), num_rows='dynamic', key='cog_hist', width='stretch',
                         column_config={DATE: st.column_config.DateColumn(DATE)})
    st.markdown('**Earlier MRI scans**')
    mri = st.data_editor(empty_mri_history(), num_rows='dynamic', key='mri_hist', width='stretch',
                         column_config={DATE: st.column_config.DateColumn(DATE),
                                        MRI_HISTORY['field']: st.column_config.SelectboxColumn(
                                            MRI_HISTORY['field'], options=FIELD_STRENGTHS)})
    return cog, mri


def _form():
    """Draw the form; returns the submitted values or None."""
    with st.form('new_patient', clear_on_submit=False):
        st.markdown('#### Patient')
        a, b, c, d = st.columns(4)
        v = dict(name=a.text_input('Patient name or ID'),
                 visit=b.date_input('Visit date', value=pd.Timestamp.today().date(), min_value=EARLIEST_DATE),
                 mci_years=c.number_input('Years since MCI was first diagnosed', 0.0, 20.0, 1.0, step=0.5),
                 sex=d.radio('Sex', ['Male', 'Female'], horizontal=True))
        a, b = st.columns(2)
        v.update(age=a.slider('Age (years)', 50, 100, 73), education=b.slider('Education (years)', 0, 25, 16))

        st.markdown('#### Cognition and daily function')
        a, b, c, d = st.columns(4)
        v.update(mmse=a.slider('MMSE (0–30)', 0, 30, 27, help='Higher is better.'),
                 faq=b.slider('FAQ (0–30)', 0, 30, 3,
                              help='Functional Activities Questionnaire. Higher means more difficulty.'),
                 adas13=c.slider('ADAS-Cog13 (0–85)', 0.0, 85.0, 17.0, step=0.5, help='Higher means more impairment.'),
                 ravlt=d.slider('Delayed word recall, RAVLT (0–15)', 0, 15, 4))

        st.markdown('#### MRI at this visit (FreeSurfer)')
        st.caption(f'Leave empty if there is no MRI from the {MRI_MAX_AGE} days before this visit.')
        a, b, c, d, e = st.columns(5)
        v.update(mri_date=a.date_input('MRI date', value=None, min_value=EARLIEST_DATE),
                 hv=b.number_input('Hippocampus L+R (mm³)', 2000.0, 14000.0, value=None, step=50.0),
                 icv=c.number_input('Intracranial vol. (mm³)', 900000.0, 2200000.0, value=None, step=10000.0),
                 entor=d.number_input('Entorhinal mean (mm)', 1.0, 6.0, value=None, step=0.05),
                 field=e.radio('Field strength', FIELD_STRENGTHS, horizontal=True))

        st.markdown('#### Earlier measurements (optional, for trends)')
        st.caption(f'Trends need at least one earlier value ≥ {MIN_SPAN} days before this visit. Earlier MRI scans '
                   'must use the same FreeSurfer version and field strength as the current scan.')
        v['cog_history'], v['mri_history'] = _history_editors()
        confirm = st.checkbox('MCI confirmed at this visit, no previous dementia diagnosis, '
                              'not on anti-amyloid therapy')
        submitted = st.form_submit_button('Save patient', type='primary')
    if not submitted:
        return None
    if not v['name'].strip() or not confirm:
        st.error('Enter a patient name or ID and confirm the eligibility statement.')
        return None
    return v


def render():
    st.title('Add a patient')
    st.caption('Enter the values from the current MCI visit. Leave MRI and history tables empty if not available; '
               'the app then uses the most complete model the data allow.')
    v = _form()
    if v is None:
        return
    name = v.pop('name').strip()
    try:
        f = manual_features(ui.resources()['prep'], **v)
    except InvalidEntry as exc:
        st.error(f'Not saved: {exc}')
        return
    day = str(pd.Timestamp(v['visit']).date())
    ui.patient_store().upsert([ProfileRow(f'Manual:{uuid.uuid4()}', name, day, json.dumps(f))])
    best = applicable_models(f)[-1]
    st.success(f'Saved **{name}**. Available model: **{ui.MODEL_INFO[best][0]}**. '
               'Open **Patient risk** to see the estimate.')
    if 'hippo_slope' in f:
        st.caption(f"Hippocampal change {f['hippo_slope']:+.0f} mm³/year over {f['mri_span_days']} days.")
    elif 'mri_span_days' in f:
        st.caption(f"MRI history spans {f['mri_span_days']} days (< {MIN_SPAN}); atrophy trend not used.")
