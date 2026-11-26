"""MCI Dementia Risk — clinical decision-support app for patients with mild cognitive impairment.

Run:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title='MCI Dementia Risk', page_icon='🧠', layout='wide')

from mci_risk.config import MIN_SPAN  # noqa: E402  (set_page_config must be the first Streamlit call)
from mci_risk.ui import add_patient, cohort, common, patient_list, patient_risk, performance  # noqa: E402

PAGES = {
    'Patient risk': patient_risk.render,
    'Add patient': add_patient.render,
    'Patient list': patient_list.render,
    'Model performance': performance.render,
    'Study cohort': cohort.render,
}
REQUIRED_RESULTS = ['preprocessing.json', 'secondary_models.json', 'train_inputs.csv']
INTENDED_USE = (
    '- Adults with a confirmed MCI diagnosis and no prior dementia diagnosis.\n'
    '- Not for patients receiving anti-amyloid antibodies (lecanemab, donanemab): treatment changes brain volume.\n'
    '- Supports follow-up planning. It does not diagnose and does not replace clinical judgement.\n'
    f'- Atrophy trends need ≥ 2 MRI scans on the same FreeSurfer version and field strength, ≥ {MIN_SPAN} days apart.\n'
    '- Developed and checked on the ADNI research cohort; performance in other populations is not yet known.')


def sidebar():
    with st.sidebar:
        st.markdown('## 🧠 MCI Dementia Risk')
        st.caption('Estimates the risk that a patient with mild cognitive impairment (MCI) will receive a dementia '
                   'diagnosis within 1–3 years.')
        page = st.radio('Navigate', list(PAGES), label_visibility='collapsed')
        with st.expander('Intended use'):
            st.markdown(INTENDED_USE)
    return page


def main():
    common.inject_css()
    page = sidebar()
    if not common.require(*REQUIRED_RESULTS):
        st.stop()
    PAGES[page]()


main()
