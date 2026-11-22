from datetime import date, timedelta
from pathlib import Path
import hashlib
import joblib
import pandas as pd
import streamlit as st
from cohort import build_frames, SETS
from storage import record, history

st.title("Hồ sơ và tái khám MCI")
st.caption("Môi trường nghiên cứu · Nhập thủ công từng đối tượng")
patient = st.text_input("Mã hồ sơ giả lập").strip()
with st.form("visit"):
    day = st.date_input("Ngày đánh giá", date.today())
    dx = st.selectbox("Chẩn đoán ghi nhận", ["MCI", "CN", "Dementia"])
    age = st.number_input("Tuổi tại lượt khám", 18, 120, value=None)
    sex = st.selectbox("Giới ghi nhận", ["Nữ", "Nam"], index=None)
    edu = st.number_input("Số năm học", 0, 40, value=None)
    mmse = st.number_input("MMSE", 0, 30, value=None)
    faq = st.number_input("FAQ", 0, 30, value=None)
    scan = st.date_input("Ngày chụp MRI", value=None)
    available = st.date_input("Ngày có kết quả MRI", value=None)
    hippo = st.number_input("Tổng thể tích hồi hải mã (mm³)", min_value=0.01, value=None)
    entor = st.number_input("Độ dày nội khứu trung bình (mm)", min_value=0.01, value=None)
    qc = st.checkbox("MRI đạt QC")
    version = st.text_input("Phiên bản xử lý MRI")
    save = st.form_submit_button("Lưu lượt khám / bản hiệu chỉnh")
if save:
    try:
        if any(x is not None for x in (hippo, entor)) and (
                not scan or not available or available < scan or not version.strip()):
            raise ValueError("MRI cần ngày chụp, ngày có kết quả hợp lệ và phiên bản xử lý.")
        record(patient, "visit", dict(date=day,dx=dx,age=age,female=None if sex is None else int(sex=="Nữ"),
            education=edu,mmse=mmse,faq=faq,scan=scan,available=available,hippo=hippo,
            entor=entor,qc=int(qc),version=version))
        st.success("Đã lưu phiên bản lượt khám.")
    except ValueError as exc:
        st.error(str(exc))
if not patient:
    st.stop()
records = history(patient)
visits = {r["payload"]["date"]:r for r in records if r["kind"]=="visit"}
st.subheader("Lịch sử hồ sơ")
st.dataframe([v["payload"] for _,v in sorted(visits.items())])
if visits:
    anchor = st.selectbox("Ngày tái khám cần đánh giá", sorted(visits, reverse=True))
    if st.button("Tạo phiếu dự báo"):
        try:
            vrows, measures, ids = [],[],[]
            for key,r in visits.items():
                if key > anchor:
                    continue
                p = r["payload"]
                ids.append(r["id"])
                vrows.append(dict(p,rid=patient,date=pd.Timestamp(key)))
                for variable in ["mmse","faq","hippo","entor"]:
                    mri = variable in ("hippo","entor")
                    if p[variable] is None or (mri and p["available"] > anchor):
                        continue
                    measures.append(dict(rid=patient,date=pd.Timestamp(p["scan"] if mri else key),
                        variable=variable,value=p[variable],qc=p["qc"] if mri else 1,
                        version=p["version"] if mri else "clinical"))
            cols = ["rid","date","variable","value","qc","version"]
            row,_ = build_frames(pd.DataFrame(vrows),pd.DataFrame(measures,columns=cols).drop_duplicates())
            if row.empty or str(row.iloc[0].t0.date()) != anchor:
                raise ValueError("Lượt chọn chưa đáp ứng mốc MCI tháng thứ sáu.")
            path = Path("runs/demo/branches.joblib")
            if not path.exists():
                raise ValueError("Chạy python branches.py demo runs/demo để tạo mô hình giả lập.")
            from train import probability
            models = joblib.load(path)
            name = next((n for n in ("M3","M2","M1","MC","M0") if n in models and
                         row[SETS[n]].notna().all(axis=None)),None)
            if name is None:
                raise ValueError("Chưa đủ dữ liệu cho các nhánh hiện có.")
            inputs=row[SETS[name]]
            risk=float(probability(models[name],inputs)[0])
            record(patient,"prediction",dict(t0=anchor,horizon=183,model=name,
                probability=risk,features=inputs.iloc[0].to_dict(),visit_versions=ids,
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source="synthetic"))
            st.success(f"{name} · Xác suất minh họa trong 183 ngày: {risk:.1%}")
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
