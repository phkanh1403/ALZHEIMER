"""Convert a staged ADNI package using a reviewed, phase-specific dictionary."""
import json
import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from cohort import build, SETS


def resolve_duplicates(visits, measures, additional_review=()):
    visits = pd.DataFrame(visits)
    measures = pd.DataFrame(measures, columns=[
        "rid", "date", "variable", "value", "qc", "version", "source", "image_uid", "processed",
    ])
    review = list(additional_review)
    visit_keys = ["rid", "date"]
    measure_keys = visit_keys + ["variable"]

    def hold(frame, action, reason):
        for row in frame.to_dict("records"):
            review.append(dict(row, action=action, reason=reason))

    def collapse(frame):
        columns = [column for column in frame.columns if column != "source"]
        duplicates = frame.duplicated(columns, keep="first")
        hold(frame.loc[duplicates], "merged", "identical_normalized_record")
        return frame.loc[~duplicates].copy()

    visits = collapse(visits)
    measures = collapse(measures)
    conflicts = visits.groupby(visit_keys)["dx"].transform("nunique").gt(1)
    excluded_rids = visits.loc[conflicts, "rid"]
    for frame in (visits, measures):
        hold(frame.loc[frame["rid"].isin(excluded_rids)], "excluded", "rid_has_conflicting_diagnosis")
    visits = visits.loc[~visits["rid"].isin(excluded_rids)].copy()
    measures = measures.loc[~measures["rid"].isin(excluded_rids)].copy()

    repeated_visit = visits.duplicated(visit_keys, keep=False)
    hold(visits.loc[repeated_visit], "merged", "same_diagnosis_review_demographics")
    for _, group in visits.loc[repeated_visit].groupby(visit_keys, sort=False):
        for column in ("age", "female", "education"):
            values = group[column].dropna().unique()
            visits.loc[group.index, column] = values[0] if len(values) == 1 else None
    visits = visits.drop_duplicates(visit_keys)

    is_mri = measures["variable"].isin({"hippo", "entor"})
    mri = measures.loc[is_mri]
    image_counts = mri.groupby(visit_keys)["image_uid"].nunique()
    repeated_mri = mri.duplicated(measure_keys, keep=False)
    ambiguous = image_counts.index[image_counts.gt(1)].union(
        pd.MultiIndex.from_frame(mri.loc[repeated_mri, visit_keys].drop_duplicates())
    )
    exclude_mri = is_mri & pd.MultiIndex.from_frame(measures[visit_keys]).isin(ambiguous)
    hold(measures.loc[exclude_mri], "excluded", "ambiguous_mri_same_day")
    measures = measures.loc[~exclude_mri].copy()

    repeated_measure = measures.duplicated(measure_keys, keep=False)
    hold(measures.loc[repeated_measure], "excluded", "conflicting_measure")
    measures = measures.loc[~repeated_measure]
    review_columns = [
        "action", "reason", "source", "rid", "date", "dx", "age", "female",
        "education", "variable", "value", "image_uid", "version", "processed", "qc",
        "PTDOBYY", "PTEDUCAT", "PTGENDER",
    ]
    return visits, measures, pd.DataFrame(review, columns=review_columns)


def convert(root, destination, rules):
    if not rules.get("reviewed") or not rules.get("phases"):
        raise ValueError("Cần xác nhận ánh xạ theo DATADIC của bản xuất trước khi chạy.")
    def read(name):
        matches=list(root.rglob(name+".csv"))
        if len(matches)!=1:
            raise ValueError("Cần đúng một tệp "+name)
        return pd.read_csv(matches[0],dtype=str,keep_default_na=False)
    def day(value):
        date = pd.to_datetime(value, errors="raise")
        if pd.isna(date):
            raise ValueError("Thiếu ngày hoặc ngày không hợp lệ")
        return date.date()
    visits, measures, issues = [],[],[]
    phase_review = []
    dem=read("PTDEMOG")
    demographic_review = {}
    dem["source"] = [f"PTDEMOG:{index + 2}" for index in dem.index]
    dem["d"] = pd.to_datetime(dem.VISDATE, errors="coerce")
    demographics = dict(tuple(dem.groupby(["RID", "PHASE"], sort=False)))
    empty_demographics = dem.iloc[:0]
    for table in ("DXSUM","MMSE","FAQ","UCSFFSX7"):
        frame = read(table)
        if table == "DXSUM":
            for phase_name in frame.PHASE.unique():
                if phase_name not in rules["phases"]:
                    continue
                rule = rules["phases"][phase_name]
                if rule.get("dx_field") not in frame.columns:
                    codes = frame.loc[frame.PHASE.eq(phase_name), "DIAGNOSIS"].value_counts().head(12).to_dict() if "DIAGNOSIS" in frame else {}
                    raise ValueError(f"DXSUM pha {phase_name}: cấu hình {rule.get('dx_field')} không khớp header. Mã DIAGNOSIS quan sát: {codes}. Đối chiếu DATADIC trước khi sửa mapping.")
        for idx,r in frame.iterrows():
            ref=f"{table}:{idx+2}"
            if r.PHASE not in rules["phases"]:
                reason = f"Pha chưa ánh xạ: {r.PHASE or '(trống)'}"
                phase_review.append({
                    "source": ref,
                    "phase": r.PHASE,
                    "rid": r.RID,
                    "date_raw": r.get("EXAMDATE") or r.get("VISDATE", ""),
                    "diagnosis_raw": r.get("DIAGNOSIS", ""),
                    "reason": reason,
                })
                issues.append((ref, reason))
                continue
            try:
                phase=rules["phases"][r.PHASE]
                if not r.RID:
                    raise ValueError("Thiếu RID")
                if table=="DXSUM":
                    if r.get("VISCODE2","").lower() in ("sc","scmri") or r.get("VISCODE","").lower()=="sc":
                        continue
                    d=day(r[phase["dx_date"]])
                    dx=phase["diagnoses"].get(r[phase["dx_field"]])
                    if dx not in ("CN","MCI","Dementia"):
                        raise ValueError("Chẩn đoán chưa giải mã")
                    candidates = demographics.get((r.RID, r.PHASE), empty_demographics)
                    candidates=candidates[candidates.d<=pd.Timestamp(d)].sort_values("d")
                    age=female=education=None
                    if not candidates.empty:
                        latest=candidates[candidates.d==candidates.d.max()]
                        years = pd.to_numeric(latest.get("PTDOBYY", pd.Series(index=latest.index, dtype=float)), errors="coerce")
                        # Accept only an explicit four-digit year in the birth-date field.
                        fallback = latest.get("PTDOB", pd.Series("", index=latest.index)).str.extract(r"(?<![0-9])((?:18|19|20)[0-9]{2})(?![0-9])", expand=False)
                        years = years.where((d.year - years).between(18, 120)).fillna(pd.to_numeric(fallback, errors="coerce"))
                        ages = d.year - years
                        educations = pd.to_numeric(latest["PTEDUCAT"], errors="coerce")
                        sexes = latest["PTGENDER"].str.strip().map(phase["female"])
                        fields = [
                            ages.where(ages.between(18, 120)),
                            sexes,
                            educations.where(educations.between(0, 40)),
                        ]
                        unique_values = [field.dropna().unique() for field in fields]
                        age, female, education = [
                            values[0] if len(values) == 1 else None for values in unique_values
                        ]
                        if any(len(values) > 1 for values in unique_values):
                            for person in latest.to_dict("records"):
                                demographic_review[person["source"]] = dict(
                                    person, rid=r.RID, date=person["d"],
                                    action="review", reason="conflicting_demographics",
                                )
                    if age is None:
                        issues.append((ref, "age: no unique valid birth year from eligible PTDEMOG"))
                    visits.append(dict(rid=r.RID,date=d,dx=dx,age=age,female=female,
                                       education=education,source=ref))
                    continue
                if table=="UCSFFSX7":
                    d=day(r.EXAMDATE)
                    if r.RUNDATE and day(r.RUNDATE)<d:
                        raise ValueError("RUNDATE trước ngày chụp")
                    if not r.IMAGEUID or not r.FSVER:
                        raise ValueError("Thiếu IMAGEUID/FSVER")
                    for var,fields,region in (("hippo",("ST29SV","ST88SV"),"HIPPOQC"),
                                              ("entor",("ST24TA","ST83TA"),"TEMPQC")):
                        if any(r[k] not in phase["accepted"][k] for k in ("STATUS","OVERALLQC",region)):
                            issues.append((ref,var+" QC/status")); continue
                        vals=pd.to_numeric([r[k] for k in fields],errors="coerce")
                        if not all(0<v<float("inf") for v in vals):
                            issues.append((ref,var+" thiếu ROI")); continue
                        measures.append(dict(rid=r.RID,date=d,variable=var,value=sum(vals)/(2 if var=="entor" else 1),
                            qc=1,version=r.FSVER,source=ref,image_uid=r.IMAGEUID,processed=r.RUNDATE))
                else:
                    var,col=("mmse","MMSCORE") if table=="MMSE" else ("faq","FAQTOTAL")
                    d=day(r.VISDATE)
                    if table=="MMSE" and r.DONE not in phase["mmse_done"]:
                        raise ValueError("MMSE chưa hoàn tất")
                    val=float(r[col])
                    if not 0<=val<=30:
                        raise ValueError("Điểm ngoài miền")
                    measures.append(dict(rid=r.RID,date=d,variable=var,value=val,qc=1,
                                         version="clinical",source=ref))
            except (ValueError,KeyError,TypeError) as exc:
                issues.append((ref,str(exc)))
    if not visits or not measures:
        raise ValueError("Không dựng được dữ liệu. Kiểm tra mã pha, DXSUM và quy tắc QC. "+str(issues[:3]))
    destination.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        out=Path(tmp)/"export";out.mkdir()
        clean_visits, clean_measures, review = resolve_duplicates(
            visits, measures, demographic_review.values()
        )
        clean_visits.to_csv(out / "visits.csv", index=False)
        clean_measures.to_csv(out / "measures.csv", index=False)
        review.to_csv(out / "etl_duplicates_review.csv", index=False)
        pd.DataFrame(issues,columns=["source","reason"]).to_csv(out/"etl_issues.csv",index=False)
        pd.DataFrame(phase_review, columns=[
            "source", "phase", "rid", "date_raw", "diagnosis_raw", "reason",
        ]).to_csv(out/"etl_unmapped_phases_review.csv", index=False)
        data,_=build(out)
        if not data.empty:
            x=data.drop(columns=["days","event"]).copy()
            for n,c in SETS.items():
                x["eligible_"+n]=x[c].notna().all(axis=1)
            x.to_csv(out/"features_m06.csv",index=False)
            y=data[["rid","t0","days","event"]].copy()
            y["status_h183"]=["event" if e and d<=183 else "no_recorded_event" if d>=183 else
                              "unknown_early_censoring" if d else "no_followup" for e,d in zip(y.event,y.days)]
            y.to_csv(out/"outcomes_h183.csv",index=False)
        destination.mkdir()
        try:
            for file in out.iterdir():
                shutil.copyfile(file, destination / file.name)
            (destination / ".ready").touch()
        except OSError:
            shutil.rmtree(destination, ignore_errors=True)
            raise


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("source","output","mapping"):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    convert(a.source,a.output,json.loads(a.mapping.read_text(encoding="utf-8")))
