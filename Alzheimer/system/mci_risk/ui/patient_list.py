"""Page: search, star and delete saved patients; import an ADNI export."""
import json
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from ..adni_import import import_zip
from ..config import OPTIONAL_TABLES, REQUIRED_TABLES
from ..partition import in_training
from ..risk import applicable_models
from . import common as ui

RESULT_KEY = 'last_import'


def _import_section(store):
    _show_last_import()
    with st.expander('Import patients from an ADNI data export (ZIP)'):
        st.caption(f'The ZIP must contain {", ".join(REQUIRED_TABLES)} CSV files ({" and ".join(OPTIONAL_TABLES)} are '
                   'optional). Each participant is imported at their most recent MCI visit; importing again adds new '
                   'visits and updates existing ones.')
        upload = st.file_uploader('ADNI export', type='zip', label_visibility='collapsed')
        if not (upload and st.button('Import')):
            return
        try:
            with st.spinner('Reading files and preparing patients…'):
                r = import_zip(store, upload, ui.resources()['prep'], in_training)   # was: set(ui.cohort_inputs().RID)
        except (BadZipFile, ValueError, KeyError, OSError) as exc:   # malformed export: report it, keep the app up
            st.error(f'Import failed: {exc}')
            return
        st.session_state[RESULT_KEY] = r      # shown after the rerun that refreshes the patient table
        st.rerun()


def import_message(r):
    """(kind, text) describing an import result; kind is 'success' or 'warning'."""
    if r['rids'] and r['skipped_train'] == r['rids']:
        return 'warning', (f"No patients added: all {r['rids']} participants in this ZIP are part of the model's "
                           'training data (80% split), and the app does not score the people the model was trained on. '
                           'Import data from other participants, for example the 20% test split (data/test).')
    return 'success', (f"Added {r['added']} and updated {r['updated']} patient visits from {r['rids']} participants. "
                       f"{r['skipped_train']} participants were skipped because they are in the model's training "
                       'data.')


def _show_last_import():
    r = st.session_state.pop(RESULT_KEY, None)
    if r is not None:
        kind, text = import_message(r)
        getattr(st, kind)(text)

def _best_model(features):
    usable = applicable_models(features)
    return ui.MODEL_INFO[usable[-1]][0] if usable else 'Incomplete'


def render():
    st.title('Patient list')
    store = ui.patient_store()
    _import_section(store)
    patients = store.all()
    if patients.empty:
        st.info('No patients yet.')
        return
    feats = patients.features.map(json.loads)
    view = pd.DataFrame({
        '★': patients.star.map({1: '★', 0: ''}).values, 'Patient': patients.name.values, 'Visit': patients.day.values,
        'Age': feats.map(lambda f: f.get('age')).values, 'MMSE': feats.map(lambda f: f.get('mmse')).values,
        'ADAS-Cog13': feats.map(lambda f: f.get('adas13')).values,
        'Latest MRI': feats.map(lambda f: f.get('mri_date')).values,
        'Best model': feats.map(_best_model).values})
    query = st.text_input('Search by name or ID', placeholder='Type to filter…')
    keep = view.Patient.str.contains(query, case=False, regex=False)
    shown = view[keep]
    k1, k2, k3 = st.columns(3)
    k1.metric('Patients', int(keep.sum()))
    k2.metric('With MRI', int(shown['Latest MRI'].notna().sum()))
    k3.metric('With atrophy trend', int((shown['Best model'] == ui.MODEL_INFO['M3'][0]).sum()))
    st.dataframe(shown.assign(**{'Latest MRI': shown['Latest MRI'].fillna('—')}), hide_index=True, width='stretch',
                 column_config={'Age': st.column_config.NumberColumn(format='%d'),
                                'ADAS-Cog13': st.column_config.NumberColumn(format='%.1f')})
    if not keep.any():
        return
    sel = st.selectbox('Select a patient', patients.index[keep.values],
                       format_func=lambda i: f"{patients.loc[i, 'name']} · {patients.loc[i, 'day']}")
    profile_id = patients.loc[sel, 'id']
    a, b, c = st.columns([1, 1, 2])
    if a.button('☆ Unstar' if patients.loc[sel, 'star'] else '★ Star'):
        store.toggle_star(profile_id)
        st.rerun()
    sure = c.checkbox('Confirm deletion')
    if b.button('Delete', disabled=not sure):
        store.delete(profile_id)
        st.rerun()
