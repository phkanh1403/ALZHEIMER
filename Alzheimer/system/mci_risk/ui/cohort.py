"""Page: the ADNI training cohort — diagnoses, outcomes, distributions and data-quality checks."""
import numpy as np
import plotly.express as px
import streamlit as st

from ..config import DAYS_PER_MONTH, HORIZONS, TRAIN_DIR, WINDOW
from ..data import DX_DEMENTIA, DX_MCI, DX_NORMAL, has_table, read_table
from . import common as ui

LONGEST = max(HORIZONS)
OUTCOME = f'Outcome within {ui.HORIZON_LABEL[LONGEST]}'
OUTCOME_COLORS = {'Diagnosed with dementia': ui.RED, 'Dementia-free': ui.GREEN,
                  f'Follow-up < {ui.HORIZON_LABEL[LONGEST]}': ui.GREY}
DIAGNOSES = {DX_NORMAL: 'Cognitively normal', DX_MCI: 'MCI', DX_DEMENTIA: 'Dementia'}
HV_RANGE = [3000, 10500]   # mm³, axis range for hippocampal volume


def _conflict_kind(action):
    if action == 'averaged':
        return 'Repeat scans averaged'
    if action.startswith('kept'):
        return 'Same-day 1.5T/3T scans: one kept'
    if action.startswith('latest record'):
        return 'Inconsistent demographics'
    if action == 'measurement removed':
        return 'Conflicting test score removed'
    return 'Conflicting diagnosis removed'


@st.cache_data
def _latest_diagnoses():
    if not has_table(TRAIN_DIR, 'DXSUM'):
        return None
    dx = read_table(TRAIN_DIR, 'DXSUM').dropna(subset=['EXAMDATE'])
    return dx[dx.DIAGNOSIS.isin(list(DIAGNOSES))].sort_values('EXAMDATE').drop_duplicates('RID', keep='last')


def _pie(where, frame, column, title, **kw):
    d = frame[column].value_counts().reset_index()
    fig = px.pie(d, names=column, values='count', hole=0.5, **kw)
    fig.update_layout(title=title, height=340, legend=dict(orientation='h', y=-0.1))
    ui.plot(fig, where)


def _overview(lm, first, latest):
    k = st.columns(4)
    k[0].metric('Participants with a diagnosis', f'{latest.RID.nunique():,}' if latest is not None else '—')
    k[1].metric('Patients with MCI visits', f'{first.RID.nunique():,}')
    k[2].metric('MCI visits analysed', f'{len(lm):,}')
    k[3].metric('Later diagnosed with dementia', f"{int(lm.groupby('RID').event.max().sum()):,}")
    a, b, c = st.columns(3)
    if latest is not None:
        _pie(a, latest.assign(Diagnosis=latest.DIAGNOSIS.map(DIAGNOSES)), 'Diagnosis', 'Most recent diagnosis',
             color='Diagnosis', color_discrete_map={'Cognitively normal': ui.GREEN, 'MCI': ui.AMBER,
                                                    'Dementia': ui.RED})
    _pie(b, first, OUTCOME, f'{OUTCOME} of the first MCI visit', color=OUTCOME, color_discrete_map=OUTCOME_COLORS)
    _pie(c, first, 'Sex', 'Sex', color_discrete_sequence=[ui.BLUE, '#9bb8e6'])
    share = first[OUTCOME].value_counts(normalize=True)
    st.caption(f'Left: the latest diagnosis of every training participant. Middle: what happened within '
               f'{ui.HORIZON_LABEL[LONGEST]} of each patient\'s first MCI visit — '
               f"{share.get('Diagnosed with dementia', 0):.0%} were diagnosed with dementia, "
               f"{share.get('Dementia-free', 0):.0%} stayed dementia-free, and the rest were followed for less "
               f'than {ui.HORIZON_LABEL[LONGEST]}. Right: sex of the patients with MCI visits.')


def _distributions(first):
    a, b = st.columns(2)
    fig = px.histogram(first, x='age', color=OUTCOME, nbins=30, barmode='stack', color_discrete_map=OUTCOME_COLORS,
                       labels={'age': 'Age at first MCI visit (years)'})
    fig.update_layout(title='Age distribution', height=360, legend=dict(orientation='h', y=-0.3, title=''),
                      yaxis_title='Patients')
    ui.plot(fig, a)
    fig = px.histogram(first, x='mmse', color=OUTCOME, barmode='group', color_discrete_map=OUTCOME_COLORS,
                       labels={'mmse': 'MMSE at first MCI visit'})
    fig.update_layout(title='MMSE distribution', height=360, legend=dict(orientation='h', y=-0.3, title=''),
                      yaxis_title='Patients')
    ui.plot(fig, b)
    med = first.groupby(OUTCOME)[['age', 'mmse']].median()
    a.caption(f"Age at the first MCI visit, coloured by later outcome. Median age: "
              f"{_by_outcome(med.age, '{:.0f}')}.")
    b.caption(f"MMSE (0–30, lower = more impaired) at the first MCI visit. Median MMSE: "
              f"{_by_outcome(med.mmse, '{:.0f}')}.")

    a, b = st.columns(2)
    sub = first.dropna(subset=['hippo_adj'])
    fig = px.violin(sub, x=OUTCOME, y='hippo_adj', color=OUTCOME, box=True, points=False,
                    color_discrete_map=OUTCOME_COLORS,
                    labels={'hippo_adj': 'Hippocampal volume, head-size adjusted (mm³)'})
    fig.update_layout(title='Hippocampal volume by later outcome', height=400, showlegend=False, xaxis_title='',
                      yaxis_range=HV_RANGE)
    ui.plot(fig, a)
    a.caption('Head-size-adjusted hippocampal volume at the first MCI visit with an MRI; the box marks the middle '
              f"half of patients. Median volume: {_by_outcome(sub.groupby(OUTCOME).hippo_adj.median(), '{:,.0f} mm³')}"
              '. Smaller volumes go with a later dementia diagnosis.')
    sub = first.dropna(subset=['hippo_adj', 'adas13'])
    fig = px.scatter(sub, x='adas13', y='hippo_adj', color=OUTCOME, opacity=0.6, color_discrete_map=OUTCOME_COLORS,
                     labels={'adas13': 'ADAS-Cog13', 'hippo_adj': 'Hippocampal volume (mm³)'})
    fig.update_layout(title='Cognitive testing and hippocampal volume', height=400,
                      legend=dict(orientation='h', y=-0.25, title=''), yaxis_range=HV_RANGE)
    ui.plot(fig, b)
    corr = sub[['adas13', 'hippo_adj']].corr().iloc[0, 1]
    b.caption(f'Each dot is one patient. Higher ADAS-Cog13 (worse cognition) goes with smaller hippocampal volume '
              f'(correlation {corr:.2f}); patients later diagnosed cluster towards the lower right, where both '
              'measures are worse.')


def _by_outcome(values, pattern):
    return '; '.join(f'{k.lower()} {pattern.format(v)}' for k, v in values.items())


def _timing_and_quality(lm):
    a, b = st.columns([1.5, 1])
    events = lm[np.isfinite(lm.R) & (lm.R <= WINDOW)]
    fig = px.histogram(events, x=events.R / DAYS_PER_MONTH, nbins=60, color_discrete_sequence=[ui.BLUE])
    fig.update_layout(title='When dementia was recorded after an MCI visit', height=360, bargap=0.05,
                      xaxis_title='Months after the MCI visit', yaxis_title='Diagnoses')
    ui.plot(fig, a)
    a.caption('Peaks at 6, 12, 18 and 24 months follow the study visit schedule: dementia is only recognised at a '
              'visit. The models account for this.')
    conflicts = ui.run_csv('conflicts.csv')
    if conflicts is None:
        return
    counts = conflicts.action.map(_conflict_kind).value_counts().rename_axis('action').reset_index(name='records')
    fig = px.pie(counts, names='action', values='records', hole=0.5,
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(title='Data quality checks applied', height=360, legend=dict(orientation='h', y=-0.2))
    ui.plot(fig, b)
    b.caption(f'{counts.records.sum():,} records were changed by the cleaning rules (none silently): each slice is '
              'one kind of rule. The log lists every record.')
    b.download_button('Download data-quality log (CSV)', conflicts.to_csv(index=False).encode('utf-8'),
                      'data_quality_log.csv')


def render():
    st.title('Study cohort')
    st.caption('ADNI participants used to build the models (training portion, 80% of participants).')
    if not ui.has_cohort_data():
        st.info('This page needs the participant-level ADNI files, which are covered by the ADNI Data Use Agreement '
                'and are not part of the distributed release. To see it, download the ADNI tables listed in '
                'README.md from LONI, split them with split_dataset.py and run train.py.')
        return
    lm = ui.cohort_landmarks()
    first = lm.sort_values('t0').drop_duplicates('RID').copy()
    first[OUTCOME] = np.where(first.R <= LONGEST, 'Diagnosed with dementia',
                              np.where(first.L >= LONGEST, 'Dementia-free', f'Follow-up < {ui.HORIZON_LABEL[LONGEST]}'))
    first['Sex'] = first.sex.map({1.0: 'Male', 2.0: 'Female'})
    _overview(lm, first, _latest_diagnoses())
    _distributions(first)
    _timing_and_quality(lm)
