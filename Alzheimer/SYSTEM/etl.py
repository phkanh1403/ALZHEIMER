"""Convert a staged ADNI package using a reviewed, phase-specific dictionary."""
import json
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
from cohort import build, SETS


def convert(root, destination, rules):
    if not rules.get("reviewed") or not rules.get("phases"):
        raise ValueError("Cần xác nhận ánh xạ theo DATADIC của bản xuất trước khi chạy.")
    def read(name):
        matches=list(root.rglob(name+".csv"))
        if len(matches)!=1:
            raise ValueError("Cần đúng một tệp "+name)
        return pd.read_csv(matches[0],dtype=str,keep_default_na=False)
    def day(value):
        return pd.to_datetime(value,errors="raise").date()
    visits, measures, issues = [],[],[]
    dem=read("PTDEMOG")
    for table in ("DXSUM","MMSE","FAQ","UCSFFSX7"):
        for idx,r in read(table).iterrows():
            ref=f"{table}:{idx+2}"
            try:
                phase=rules["phases"].get(r.PHASE)
                if not phase:
                    raise ValueError("Pha chưa ánh xạ")
                if not r.RID:
                    raise ValueError("Thiếu RID")
                if table=="DXSUM":
                    if r.get("VISCODE2","").lower() in ("sc","scmri") or r.get("VISCODE","").lower()=="sc":
                        continue
                    d=day(r[phase["dx_date"]])
                    dx=phase["diagnoses"].get(r[phase["dx_field"]])
                    if dx not in ("CN","MCI","Dementia"):
                        raise ValueError("Chẩn đoán chưa giải mã")
                    candidates=dem[(dem.RID==r.RID)&(dem.PHASE==r.PHASE)].copy()
                    candidates["d"]=pd.to_datetime(candidates.VISDATE,errors="coerce")
                    candidates=candidates[candidates.d<=pd.Timestamp(d)].sort_values("d")
                    age=female=education=None
                    if not candidates.empty:
                        latest=candidates[candidates.d==candidates.d.max()]
                        if len(latest[["PTDOBYY","PTEDUCAT","PTGENDER"]].drop_duplicates())>1:
                            raise ValueError("Nhân khẩu học mâu thuẫn")
                        x=latest.iloc[-1]
                        year=pd.to_numeric(x.PTDOBYY,errors="coerce")
                        age=d.year-year if 18<=d.year-year<=120 else None
                        edu=pd.to_numeric(x.PTEDUCAT,errors="coerce")
                        education=edu if 0<=edu<=40 else None
                        female=phase["female"].get(x.PTGENDER)
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
        for name,rows,keys in (("visits",visits,["rid","date"]),("measures",measures,["rid","date","variable"])):
            df=pd.DataFrame(rows)
            duplicates=df.duplicated(keys,keep=False)
            if duplicates.any():
                raise ValueError("Khóa trùng cần đối soát: "+str(df.loc[duplicates,"source"].head().tolist()))
            df.to_csv(out/(name+".csv"),index=False)
        pd.DataFrame(issues,columns=["source","reason"]).to_csv(out/"etl_issues.csv",index=False)
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
        if destination.exists():
            raise ValueError("Thư mục kết quả đã tồn tại. Chọn phiên bản mới.")
        out.rename(destination)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("source","output","mapping"):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    convert(a.source,a.output,json.loads(a.mapping.read_text(encoding="utf-8")))
