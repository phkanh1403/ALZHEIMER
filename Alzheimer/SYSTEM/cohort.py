from pathlib import Path
import pandas as pd
import numpy as np

BASE = ["age", "female", "education", "mmse", "faq"]
MRI = ["hippo", "entor"]
SETS = {"M0": BASE, "M1": BASE + MRI, "M2": BASE + MRI + ["mmse_rate", "faq_rate"]}
SETS["M3"] = SETS["M2"] + ["hippo_rate", "entor_rate"]
SETS["MC"] = BASE + ["mmse_rate","faq_rate"]

def build(folder):
    """Read canonical tables. See README for units and source mappings."""
    folder = Path(folder)
    def read(name):
        frame = pd.read_csv(folder / (name + ".csv"), dtype={"rid": str})
        frame["date"] = pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
        if frame[["rid","date"]].isna().any().any():
            raise ValueError(f"Missing key in {name}")
        return frame.drop_duplicates()
    return build_frames(read("visits"), read("measures"))

def build_frames(visits, measures):
    if visits.duplicated(["rid","date"]).any() or measures.duplicated(["rid","date","variable"]).any():
        raise ValueError("Conflicting records at the same date. Resolve before analysis.")
    if not visits.dx.isin(["MCI","CN","Dementia"]).all():
        raise ValueError("Unmapped diagnosis")
    rows, audit = [], []
    for rid, visits_rid in visits.groupby("rid"):
        v = visits_rid.sort_values("date")
        initial = v[v.dx.eq("MCI")]
        if initial.empty:
            continue
        first = initial.iloc[0].date
        target = first + pd.DateOffset(months=6)
        distance = (v.date-target).abs().dt.days
        candidates = v[distance.le(30)]
        if candidates.empty:
            continue
        index = distance.loc[candidates.index].idxmin()
        visit = v.loc[index]
        t0 = visit.date
        if visit.dx != "MCI" or v[v.date.le(t0)].dx.eq("Dementia").any():
            continue
        row = dict(rid=rid, t0=t0, age=visit.age, female=visit.female, education=visit.education)
        measurements = measures[measures.rid.eq(rid) & measures.qc.eq(1)]
        for variable in ["mmse","faq",*MRI]:
            data = measurements[measurements.variable.eq(variable)].sort_values("date")
            if variable in MRI:
                data = data[data.value.gt(0)]
            def preceding(anchor):
                window = data[data.date.between(anchor-pd.Timedelta(days=45), anchor)]
                if "processed" in window:
                    window = window[pd.to_datetime(window.processed,errors="coerce").fillna(anchor).le(anchor)]
                return None if window.empty else window.iloc[-1]
            a,b = preceding(first), preceding(t0)
            for label, record in (("previous", a), ("current", b)):
                if record is not None:
                    row[f"{variable}_{label}_date"] = record.date
                    row[f"{variable}_{label}_source"] = record.get("source", "")
            row[variable] = np.nan if b is None else b.value
            rate = np.nan
            if a is not None and b is not None:
                days = (b.date-a.date).days
                if days >= 90 and (variable not in MRI or a.version == b.version):
                    rate = (b.value-a.value)*365.25/days
            row[variable+"_rate"] = rate
        future = v[v.date.gt(t0)]
        events = future[future.dx.eq("Dementia")]
        row["event"] = int(not events.empty)
        end = events.iloc[0].date if row["event"] else v.iloc[-1].date
        row["days"] = (end-t0).days
        rows.append(row)
        audit.append(dict(rid=rid, first=first, t0=t0, end=end))
    return pd.DataFrame(rows), pd.DataFrame(audit)
