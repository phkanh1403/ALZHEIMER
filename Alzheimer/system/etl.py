"""Chuẩn hóa dữ liệu ADNI và lưu các bản ghi cần đối soát."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from cohort import SETS, build


VISIT_KEYS = ["rid", "date"]
MEASURE_KEYS = ["rid", "date", "variable"]
DEMOGRAPHIC_FIELDS = ["age", "female", "education"]
MRI_VARIABLES = {"hippo", "entor"}
HORIZON_DAYS = 183

REVIEW_COLUMNS = [
    "action", "reason", "source", "rid", "date", "dx", "age", "female",
    "education", "variable", "value", "image_uid", "version", "processed",
    "qc", "PTDOBYY", "PTEDUCAT", "PTGENDER",
]


def read_table(root, name):
    matches = list(root.rglob(f"{name}.csv"))
    if len(matches) != 1:
        raise ValueError(f"Cần đúng một tệp {name}.csv.")
    return pd.read_csv(matches[0], dtype=str, keep_default_na=False)


def parse_day(value):
    parsed = pd.to_datetime(value, errors="raise")
    if pd.isna(parsed):
        raise ValueError("Thiếu ngày hoặc ngày không hợp lệ.")
    return parsed.date()


def add_review(review, frame, action, reason):
    for row in frame.to_dict("records"):
        review.append(dict(row, action=action, reason=reason))


def collapse_identical(frame, review):
    columns = [column for column in frame.columns if column != "source"]
    repeated = frame.duplicated(subset=columns, keep="first")
    add_review(review, frame.loc[repeated], "merged", "identical_normalized_record")
    return frame.loc[~repeated].copy()


def resolve_duplicates(visits, measures, demographic_review):
    review = list(demographic_review)
    visits = collapse_identical(pd.DataFrame(visits), review)
    measures = collapse_identical(pd.DataFrame(measures), review)

    # Một chẩn đoán mâu thuẫn có thể thay đổi cả mốc và nhãn kết cục.
    conflicting_dx = visits.groupby(VISIT_KEYS)["dx"].transform("nunique").gt(1)
    excluded_rids = set(visits.loc[conflicting_dx, "rid"])

    excluded_visits = visits["rid"].isin(excluded_rids)
    excluded_measures = measures["rid"].isin(excluded_rids)
    reason = "rid_has_conflicting_diagnosis"
    add_review(review, visits.loc[excluded_visits], "excluded", reason)
    add_review(review, measures.loc[excluded_measures], "excluded", reason)
    visits = visits.loc[~excluded_visits].copy()
    measures = measures.loc[~excluded_measures].copy()

    # Cùng chẩn đoán trong ngày: hợp nhất các trường nhân khẩu học.
    repeated_visits = visits.duplicated(VISIT_KEYS, keep=False)
    groups = visits.loc[repeated_visits].groupby(VISIT_KEYS, sort=False)
    for _, group in groups:
        add_review(review, group, "merged", "same_diagnosis_review_demographics")
        for field in DEMOGRAPHIC_FIELDS:
            values = group[field].dropna().unique()
            visits.loc[group.index, field] = values[0] if len(values) == 1 else None
    visits = visits.drop_duplicates(VISIT_KEYS)

    # Đánh dấu cả ngày MRI nếu có nhiều ảnh hoặc nhiều kết quả cho cùng biến.
    is_mri = measures["variable"].isin(MRI_VARIABLES)
    mri = measures.loc[is_mri]
    image_counts = mri.groupby(VISIT_KEYS)["image_uid"].nunique()
    repeated_mri = mri.duplicated(MEASURE_KEYS, keep=False)
    repeated_mri_days = pd.MultiIndex.from_frame(
        mri.loc[repeated_mri, VISIT_KEYS].drop_duplicates()
    )
    ambiguous_days = image_counts.index[image_counts.gt(1)].union(repeated_mri_days)
    measure_days = pd.MultiIndex.from_frame(measures[VISIT_KEYS])
    exclude_mri = is_mri & measure_days.isin(ambiguous_days)

    add_review(review, measures.loc[exclude_mri], "excluded", "ambiguous_mri_same_day")
    measures = measures.loc[~exclude_mri].copy()

    # Các khóa còn trùng ở đây là điểm lâm sàng mâu thuẫn.
    repeated_measure = measures.duplicated(MEASURE_KEYS, keep=False)
    add_review(review, measures.loc[repeated_measure], "excluded", "conflicting_measure")
    measures = measures.loc[~repeated_measure].copy()

    return (
        visits.sort_values(VISIT_KEYS),
        measures.sort_values(MEASURE_KEYS),
        pd.DataFrame(review, columns=REVIEW_COLUMNS),
    )


def normalize_visit(row, rid, phase, demographics, empty_demographics,
                    source, demographic_review):
    visit_codes = {
        row.get("VISCODE", "").strip().lower(),
        row.get("VISCODE2", "").strip().lower(),
    }
    if visit_codes.intersection({"sc", "scmri"}):
        return None

    date = parse_day(row[phase["dx_date"]])
    diagnosis = phase["diagnoses"].get(row[phase["dx_field"]].strip())
    if diagnosis not in {"CN", "MCI", "Dementia"}:
        raise ValueError("Chẩn đoán chưa giải mã.")

    visit = dict(
        rid=rid, date=date, dx=diagnosis, age=None, female=None,
        education=None, source=source,
    )
    candidates = demographics.get((rid, row.PHASE.strip()), empty_demographics)
    candidates = candidates.loc[candidates["d"] <= pd.Timestamp(date)]
    if candidates.empty:
        return visit

    latest = candidates.loc[candidates["d"] == candidates["d"].max()]
    ages = date.year - pd.to_numeric(latest["PTDOBYY"], errors="coerce")
    education = pd.to_numeric(latest["PTEDUCAT"], errors="coerce")
    fields = {
        "age": ages.where(ages.between(18, 120)),
        "female": latest["PTGENDER"].str.strip().map(phase["female"]),
        "education": education.where(education.between(0, 40)),
    }

    has_conflict = False
    for field, series in fields.items():
        values = series.dropna().unique()
        if len(values) == 1:
            visit[field] = values[0]
        elif len(values) > 1:
            has_conflict = True

    if has_conflict:
        for person in latest.to_dict("records"):
            demographic_review[person["source"]] = dict(
                person, rid=rid, date=person["d"].date(),
                action="review", reason="conflicting_demographics",
            )

    return visit


def normalize_mri(row, rid, phase, source, issues):
    date = parse_day(row.EXAMDATE)
    run_date = row.RUNDATE.strip()
    processed = parse_day(run_date) if run_date else None
    if processed is not None and processed < date:
        raise ValueError("RUNDATE trước ngày chụp.")

    image_uid = row.IMAGEUID.strip()
    version = row.FSVER.strip()
    if not image_uid or not version:
        raise ValueError("Thiếu IMAGEUID hoặc FSVER.")

    measures = []
    regions = (
        ("hippo", ("ST29SV", "ST88SV"), "HIPPOQC", 1),
        ("entor", ("ST24TA", "ST83TA"), "TEMPQC", 2),
    )
    for variable, fields, region, divisor in regions:
        qc_fields = ("STATUS", "OVERALLQC", region)
        if any(row[key].strip() not in phase["accepted"][key] for key in qc_fields):
            issues.append((source, f"{variable}: không đạt quy tắc QC/status"))
            continue

        values = pd.to_numeric([row[key] for key in fields], errors="coerce")
        if not all(0 < value < float("inf") for value in values):
            issues.append((source, f"{variable}: thiếu ROI hợp lệ"))
            continue

        measures.append(dict(
            rid=rid, date=date, variable=variable, value=sum(values) / divisor,
            qc=1, version=version, source=source,
            image_uid=image_uid, processed=processed,
        ))

    return measures


def normalize_clinical_measure(row, rid, table, phase, source):
    variable, column = (
        ("mmse", "MMSCORE") if table == "MMSE" else ("faq", "FAQTOTAL")
    )
    date = parse_day(row.VISDATE)
    if table == "MMSE" and row["DONE"].strip() not in phase["mmse_done"]:
        raise ValueError("MMSE chưa hoàn tất.")

    value = float(row[column])
    if not 0 <= value <= 30:
        raise ValueError("Điểm ngoài miền.")

    return dict(
        rid=rid, date=date, variable=variable, value=value,
        qc=1, version="clinical", source=source, image_uid=None, processed=None,
    )


def write_cohort_outputs(output):
    data, _ = build(output)
    if data.empty:
        return

    features = data.drop(columns=["days", "event"]).copy()
    for name, columns in SETS.items():
        features[f"eligible_{name}"] = features[columns].notna().all(axis=1)
    features.to_csv(output / "features_m06.csv", index=False)

    outcomes = data[["rid", "t0", "days", "event"]].copy()
    statuses = []
    for event, days in zip(outcomes["event"], outcomes["days"]):
        if event and days <= HORIZON_DAYS:
            statuses.append("event")
        elif days >= HORIZON_DAYS:
            statuses.append("no_recorded_event")
        elif days > 0:
            statuses.append("unknown_early_censoring")
        else:
            statuses.append("no_followup")

    outcomes["status_h183"] = statuses
    outcomes.to_csv(output / "outcomes_h183.csv", index=False)


def convert(root, destination, rules):
    phases = rules.get("phases")
    if not rules.get("reviewed") or not phases:
        raise ValueError("Cần xác nhận ánh xạ theo DATADIC trước khi chạy.")

    root = Path(root)
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Thư mục kết quả đã tồn tại: {destination}")

    visits, measures, issues = [], [], []
    demographic_review = {}

    people = read_table(root, "PTDEMOG")
    people["RID"] = people["RID"].str.strip()
    people["PHASE"] = people["PHASE"].str.strip()
    people["d"] = pd.to_datetime(people["VISDATE"], errors="coerce").dt.normalize()
    people["source"] = [f"PTDEMOG:{index + 2}" for index in people.index]
    demographics = dict(tuple(people.groupby(["RID", "PHASE"], sort=False)))
    empty_demographics = people.iloc[:0]

    for table in ("DXSUM", "MMSE", "FAQ", "UCSFFSX7"):
        for index, row in read_table(root, table).iterrows():
            source = f"{table}:{index + 2}"
            try:
                rid = row.RID.strip()
                phase = phases.get(row.PHASE.strip())
                if not rid:
                    raise ValueError("Thiếu RID.")
                if not phase:
                    raise ValueError(f"Pha chưa ánh xạ: {row.PHASE}")

                if table == "DXSUM":
                    visit = normalize_visit(
                        row, rid, phase, demographics, empty_demographics,
                        source, demographic_review,
                    )
                    if visit is not None:
                        visits.append(visit)
                elif table == "UCSFFSX7":
                    measures.extend(normalize_mri(row, rid, phase, source, issues))
                else:
                    measures.append(
                        normalize_clinical_measure(row, rid, table, phase, source)
                    )
            except (ValueError, KeyError, TypeError) as error:
                issues.append((source, str(error)))

    if not visits or not measures:
        raise ValueError(
            "Không dựng được lượt khám hoặc phép đo. "
            f"Một số dòng cần kiểm tra: {issues[:5]}"
        )

    clean_visits, clean_measures, review = resolve_duplicates(
        visits, measures, demographic_review.values()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=destination.parent) as temporary:
        output = Path(temporary)
        clean_visits.to_csv(output / "visits.csv", index=False)
        clean_measures.to_csv(output / "measures.csv", index=False)
        review.to_csv(output / "etl_duplicates_review.csv", index=False)
        pd.DataFrame(issues, columns=["source", "reason"]).to_csv(
            output / "etl_issues.csv", index=False
        )

        if not clean_visits.empty:
            write_cohort_outputs(output)

        # Công bố kết quả bằng cách chép từng tệp, rồi đánh dấu hoàn tất.
        destination.mkdir()
        try:
            for file in output.iterdir():
                shutil.copyfile(file, destination / file.name)
            (destination / ".ready").touch()
        except OSError:
            shutil.rmtree(destination, ignore_errors=True)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("mapping", type=Path)
    args = parser.parse_args()

    convert(
        args.source,
        args.output,
        json.loads(args.mapping.read_text(encoding="utf-8-sig")),
    )