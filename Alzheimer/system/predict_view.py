from datetime import date, timedelta
from pathlib import Path
import hashlib
import joblib
import pandas as pd
import streamlit as st
from cohort import build_frames, SETS
from storage import record, history

from workflow import select_patient, run_score
patient=select_patient()
if not patient:
    st.info("Chưa có hồ sơ.")
    st.stop()
records = history(patient)
visits = {r["payload"]["date"]:r for r in records if r["kind"]=="visit"}
st.subheader("Lịch sử hồ sơ")
st.dataframe([v["payload"] for _,v in sorted(visits.items())])
if visits:
    anchor = st.selectbox("Ngày tái khám cần đánh giá", sorted(visits, reverse=True))
    if st.button("Tạo phiếu dự báo"):
        try:
            ids = [r["id"] for k,r in visits.items() if k <= anchor]
            source = visits[anchor]["payload"].get("normalized_dir")
            if source:
                feature_file = Path(source)/"features_m06.csv"
                if not feature_file.is_file():
                    raise ValueError("Gói này chưa có hồ sơ đạt mốc M06.")
                imported = pd.read_csv(feature_file, dtype={"rid": str}, parse_dates=["t0"])
                row = imported[imported.rid.eq(patient.removeprefix("ADNI-")) & imported.t0.eq(pd.Timestamp(anchor))]
            else:
                vrows, measures, ids = [],[],[]
                for key,r in visits.items():
                    if key > anchor:
                        continue
                    p = r["payload"]
                    ids.append(r["id"])
                    vrows.append(dict(p,rid=patient,date=pd.Timestamp(key)))
                    for variable in ["mmse","faq","hippo","entor"]:
                        mri = variable in ("hippo","entor")
                        if p[variable] is None or (mri and (not p.get("available") or p["available"] > anchor)):
                            continue
                        measures.append(dict(rid=patient,date=pd.Timestamp(p["scan"] if mri else key),
                            variable=variable,value=p[variable],qc=p["qc"] if mri else 1,
                            version=p["version"] if mri else "clinical"))
                cols = ["rid","date","variable","value","qc","version"]
                row,_ = build_frames(pd.DataFrame(vrows),pd.DataFrame(measures,columns=cols).drop_duplicates())
            if row.empty or str(row.iloc[0].t0.date()) != anchor:
                raise ValueError("Lượt chọn chưa đáp ứng mốc MCI tháng thứ sáu.")
            result=run_score(row)
            if result is None:
                raise ValueError("Chưa đủ đầu vào cho các mô hình hiện có.")
            record(patient,"prediction",dict(result,t0=anchor,horizon=183,visit_versions=ids))
            st.success(f"{result['model']} · Nguy cơ trong 183 ngày {result['probability']:.1%}")
        except ValueError as exc:
            st.warning(str(exc))
for r in history(patient):
    if r["kind"]=="prediction":
        p=r["payload"]
        with st.expander(f"Phiếu {p['t0']} · {p['model']} · {r['created'][:19]}"):
            st.json(p)
            end=(date.fromisoformat(p["t0"])+timedelta(days=183)).isoformat()
            future=[(d,v["payload"]["dx"]) for d,v in visits.items() if d>p["t0"]]
            event=any(d<=end and dx=="Dementia" for d,dx in future)
            st.write("Đã ghi nhận Dementia trong hạn" if event else
                "Đủ theo dõi, chưa ghi nhận trong hạn" if any(d>=end for d,_ in future)
                else "Chưa đủ theo dõi")
