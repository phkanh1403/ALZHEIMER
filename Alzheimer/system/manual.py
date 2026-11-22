"""Translate user-entered research demo measurements into model features."""

import pandas as pd
from cohort import build_frames, SETS


def make_features(age, female, education, first, followup, visits_data, imaging):
    first, followup = pd.Timestamp(first), pd.Timestamp(followup)
    if abs((followup - (first + pd.DateOffset(months=6))).days) > 30:
        raise ValueError("Ngày tái khám phải nằm trong khoảng ±30 ngày quanh mốc 6 tháng.")
    if followup <= first:
        raise ValueError("Ngày tái khám phải sau ngày khám đầu.")

    visits = pd.DataFrame([
        dict(rid="INPUT", date=date, dx="MCI", age=age, female=female, education=education)
        for date in (first, followup)
    ])
    measurements = []
    for date, mmse, faq in visits_data:
        for variable, value in (("mmse", mmse), ("faq", faq)):
            measurements.append(dict(rid="INPUT", date=pd.Timestamp(date),
                                     variable=variable, value=value, qc=1, version="clinical"))
    for anchor, date, hippo, entor, qc, version in imaging:
        date = pd.Timestamp(date)
        age_days = (pd.Timestamp(anchor) - date).days
        if not 0 <= age_days <= 45:
            raise ValueError("Ngày MRI phải nằm trong 45 ngày tính lùi từ mốc khám tương ứng.")
        if not version.strip():
            raise ValueError("Nhập phiên bản xử lý MRI cho từng lần chụp.")
        for variable, value in (("hippo", hippo), ("entor", entor)):
            measurements.append(dict(rid="INPUT", date=date, variable=variable,
                                     value=value, qc=int(qc), version=version.strip()))
    row, _ = build_frames(visits, pd.DataFrame(measurements))
    if row.empty:
        raise ValueError("Hồ sơ chưa đủ điều kiện tại mốc dự báo.")
    return row.drop(columns=["event", "days"])


def missing_features(row):
    return {
        name: [col for col in SETS[name] if pd.isna(row.iloc[0][col])]
        for name in ("M2", "M3")
    }
