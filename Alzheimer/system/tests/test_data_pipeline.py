"""Regression tests on the real training data: landmarks, labels, leakage and conflict handling."""
import numpy as np

from mci_risk.config import MRI_MAX_AGE, WINDOW
from mci_risk.data import DX_MCI, raw_points


def test_latest_mci_visit_is_a_prediction_input(src, points):
    last = src.vis.groupby('RID').tail(1)
    last_mci = last[(last.dx == DX_MCI) & ~last.screen].iloc[0]
    mine = points[points.RID == last_mci.RID]
    assert mine.t0.max() == last_mci.day
    assert mine.has_followup.iloc[-1] == 0


def test_latest_visit_has_no_outcome_label(src, points, labelled):
    last = src.vis.groupby('RID').tail(1)
    last_mci = last[(last.dx == DX_MCI) & ~last.screen].iloc[0]
    assert not ((labelled.RID == last_mci.RID) & (labelled.t0 == last_mci.day)).any()


def test_labels_are_valid_intervals(labelled):
    assert (labelled.L >= 0).all()
    assert (labelled.L < labelled.R).all()
    assert (labelled.L <= WINDOW).all()
    assert (labelled.event == np.isfinite(labelled.R)).all()


def test_intervals_crossing_day_1095_keep_their_right_bound(labelled):
    assert ((labelled.L < 1095) & (labelled.R > 1095) & np.isfinite(labelled.R)).any()


def test_held_out_participants_are_excluded_from_parameters(src):
    all_rids = set(src.vis.RID)
    held = set(list(src.series_stats['adas13'])[:100])
    full, subset = src.fit_params(all_rids), src.fit_params(all_rids - held)
    assert full['adas13']['n'] - subset['adas13']['n'] == len(held)


def test_conflicts_are_logged(src):
    assert len(src.conflicts) > 0
    assert {'table', 'RID', 'day', 'detail', 'action'} <= set(src.conflicts[0])


def test_one_mri_row_per_participant_day(src):
    assert not src.scans.duplicated(['RID', 'day']).any()


def test_education_uses_the_record_on_or_before_the_visit(src, points):
    for rid in sorted(src.demo_conflict):
        d = src._demo_by[rid].dropna(subset=['education', 'day']).sort_values('day')
        for t in points[points.RID == rid].t0:
            before, after = d[d.day <= t], d[d.day > t]
            if len(before) and len(after) and after.education.iloc[-1] != before.education.iloc[-1]:
                r = raw_points(src, points[(points.RID == rid) & (points.t0 == t)])
                assert r.education.iloc[0] == before.education.iloc[-1]
                return
    raise AssertionError('no participant with a later, different education record')


def test_features_use_only_data_up_to_the_visit(src, labelled):
    raw = raw_points(src, labelled.head(400))
    for col in ['mmse_date', 'adas13_date', 'mri_date']:
        dated = raw[col].notna()
        assert (raw.loc[dated, col] <= raw.loc[dated, 't0']).all()
    assert (raw.mri_age_days.dropna() <= MRI_MAX_AGE).all()
