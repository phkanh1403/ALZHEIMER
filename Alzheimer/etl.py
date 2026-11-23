"""Import only declared CSVs. No ZIP member is extracted to disk."""
import hashlib
import io
import json
import re
from functools import lru_cache
from pathlib import PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd

PATHS = {n: f"Clinical_Data/{n}.csv" for n in ("DXSUM", "PTDEMOG", "MMSE", "FAQ")}
PATHS.update(DATADIC="Metadata/DATADIC.csv", UCSFFSX7="MRI_Data/MRI_Analysis/UCSFFSX7.csv")
FIELDS = {
    "DXSUM": "RID PHASE", "PTDEMOG": "RID PHASE VISDATE",
    "MMSE": "RID PHASE VISDATE DONE MMSCORE", "FAQ": "RID PHASE VISDATE FAQTOTAL",
    "UCSFFSX7": "RID PHASE EXAMDATE RUNDATE IMAGEUID FSVER STATUS OVERALLQC HIPPOQC TEMPQC ST29SV ST88SV ST24TA ST83TA",
    "DATADIC": "PHASE TBLNAME FLDNAME CODE TEXT UNITS",
}
@lru_cache(maxsize=32768)
def date(s):
    # Ambiguous local day/month ordering is deliberately not inferred.
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            parsed = pd.to_datetime(s, format=fmt, errors="raise")
        except (ValueError, TypeError, AttributeError):
            continue
        if pd.isna(parsed):
            raise ValueError(f"Ngày bị thiếu: {s!r}")
        return parsed.date().isoformat()
    raise ValueError(f"Ngày chưa xác minh: {s!r}")

def number(s, lo, hi):
    v = float(s)
    if not np.isfinite(v) or not lo <= v <= hi:
        raise ValueError(f"Giá trị ngoài miền: {s}")
    return v

def read_package(raw):
    if len(raw) > 200 * 1024**2:
        raise ValueError("ZIP vượt 200 MiB")
    frames = {}
    with ZipFile(io.BytesIO(raw)) as z:
        if len(z.infolist()) > 5000 or sum(i.file_size for i in z.infolist()) > 1024**3:
            raise ValueError("Gói quá lớn sau giải nén")
        for i in z.infolist():
            p = PurePosixPath(i.filename.replace("\\", "/"))
            if p.is_absolute() or ".." in p.parts:
                raise ValueError("Đường dẫn ZIP không hợp lệ")
            for name, suffix in PATHS.items():
                if not (str(p) == suffix or str(p).endswith("/" + suffix)):
                    continue
                if name in frames or i.file_size > 250 * 1024**2:
                    raise ValueError(f"Bảng trùng hoặc quá lớn: {name}")
                data = z.read(i).decode("utf-8-sig")
                import csv
                sep = csv.Sniffer().sniff(data.splitlines()[0], delimiters=",;\t").delimiter
                header = [s.strip().upper() for s in next(csv.reader(io.StringIO(data), delimiter=sep))]
                if len(set(header)) != len(header):
                    raise ValueError(f"{name}: trùng header")
                df = pd.read_csv(io.StringIO(data), sep=sep, dtype=str, keep_default_na=False)
                df.columns = header
                frames[name] = df.apply(lambda col: col.str.strip())
    if not {"DATADIC", "DXSUM"} <= frames.keys():
        raise ValueError("Cần Metadata/DATADIC.csv và Clinical_Data/DXSUM.csv")
    return frames

def normalize(frames, rules):
    if rules.get("reviewed") is not True or not isinstance(rules.get("phases"), dict):
        raise ValueError("Cần mapping đã đối soát DATADIC, reviewed=true")
    rows, issues = [], []
    for table, df in frames.items():
        missing = set(FIELDS[table].split()) - set(df.columns)
        if missing:
            if table in ("DATADIC", "DXSUM"):
                raise ValueError(f"{table} thiếu {sorted(missing)}")
            issues.append(dict(table=table, reason=f"Thiếu header {sorted(missing)}"))
            continue
        if table == "DATADIC":
            continue
        for i, r in enumerate(df.to_dict("records"), 2):
            base = dict(rid=r["RID"], phase=r["PHASE"], source=f"{table}:{i}", ptid=r.get("PTID",""))
            def review(reason):
                issues.append(dict(**base, table=table, reason=str(reason), raw=json.dumps({k:r[k] for k in (FIELDS[table].split() if table=="UCSFFSX7" else r)},ensure_ascii=False)))
            def add(kind, when, val, available=None, version="", uid=""):
                rows.append(dict(**base, kind=kind, date=when, value=val, available=available or when, version=version, uid=uid))
            try:
                if not base["rid"]:
                    raise ValueError("Thiếu RID")
                rule = rules["phases"].get(r["PHASE"], {})
                if table == "DXSUM":
                    if any(r.get(k,"").lower() in ("sc","scmri") for k in ("VISCODE","VISCODE2")):
                        continue
                    when = date(r.get(rule.get("dx_date","EXAMDATE"),""))
                    if r.get(rule.get("dx_field")) in rule.get("no_diagnosis", []):
                        review("TEAM_NODX: lượt không đánh giá chẩn đoán")
                        continue
                    dx = rule.get("diagnoses",{}).get(r.get(rule.get("dx_field"),""))
                    if dx not in ("CN","MCI","Dementia"):
                        raise ValueError("Chẩn đoán chưa giải mã")
                    add("dx",when,dx)
                elif table == "UCSFFSX7":
                    when, ready = date(r["EXAMDATE"]), date(r["RUNDATE"])
                    if ready < when or not r["FSVER"] or not r["IMAGEUID"]:
                        raise ValueError("Ngày xử lý, UID hoặc phiên bản MRI không hợp lệ")
                    for var,a,b,q in [("hippo","ST29SV","ST88SV","HIPPOQC"),("entor","ST24TA","ST83TA","TEMPQC")]:
                        try:
                            if not all(r[k] in rule.get("accepted",{}).get(k,[]) for k in ("STATUS","OVERALLQC",q)):
                                raise ValueError("QC chưa đạt/chưa ánh xạ")
                            x,y = (number(r[k],1e-9,1e9) for k in (a,b))
                            add(var,when,(x+y)/(2 if var=="entor" else 1),ready,r["FSVER"],r["IMAGEUID"])
                        except ValueError as exc:
                            review(f"{var}: {exc}")
                else:
                    when = date(r["VISDATE"])
                    if table == "PTDEMOG":
                        for col,var,lo,hi in [("PTDOBYY","birthyear",1800,2100),("PTEDUCAT","education",0,40)]:
                            try:
                                raw_value = r.get(col,"")
                                if col == "PTDOBYY" and re.fullmatch(r"\d{4}-\d{2}-\d{2}",raw_value):
                                    raw_value = date(raw_value)[:4]
                                add(var,when,number(raw_value,lo,hi))
                            except ValueError as exc:
                                review(f"{col}: {exc}")
                        female=rule.get("female",{}).get(r.get("PTGENDER"))
                        if female in (0,1): add("female",when,female)
                        else: review("PTGENDER chưa giải mã")
                    elif table=="FAQ":
                        add("faq",when,number(r["FAQTOTAL"],0,30))
                    elif r["DONE"] in rule.get("mmse_done",[]):
                        add("mmse",when,number(r["MMSCORE"],0,30))
                    else:
                        review("MMSE chưa xác minh hoàn tất")
            except (ValueError,KeyError,TypeError) as exc:
                review(exc)
                if table=="DXSUM" and base["rid"]:
                    try:
                        when=date(r.get(rule.get("dx_date","EXAMDATE"),""))
                    except ValueError:
                        when="0001-01-01"  # unknown ordering: block this RID
                    add("dx",when,"Unknown")
    return rows,issues

def digest(raw):
    return hashlib.sha256(raw).hexdigest()
