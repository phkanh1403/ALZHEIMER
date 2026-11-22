from datetime import date, timedelta
from pathlib import Path
import hashlib
import joblib
import pandas as pd
import streamlit as st
from cohort import build_frames, SETS
from storage import record, history

patient = st.text_input("Mã hồ sơ").strip()
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

with st.expander("Thử bằng hồ sơ mẫu"):
    rid=st.selectbox("Chọn hồ sơ mẫu",pd.read_csv("demo/visits.csv").rid.unique())
    if st.button("Lưu hồ sơ mẫu và dự đoán"):
        from workflow import run_score
        from cohort import build
        visits=pd.read_csv("demo/visits.csv").query("rid==@rid")
        measures=pd.read_csv("demo/measures.csv").query("rid==@rid")
        previous=history(rid)
        if previous:
            st.info("Mã hồ sơ đã được lưu.")
        else:
            data,_=build("demo")
            try:
                result=run_score(data[data.rid==rid])
                for v in visits.to_dict("records"):
                    values=measures[measures.date==v["date"]].set_index("variable").value
                    payload=dict(v,scan=v["date"],available=v["date"],version="demo-v1",qc=1)
                    for var in ("mmse","faq","hippo","entor"):
                        payload[var]=values.get(var,None)
                    record(rid,"visit",payload)
                if result:
                    record(rid,"prediction",dict(result,t0=str(data[data.rid==rid].iloc[0].t0.date()),horizon=183))
                st.success("Đã lưu vào Hồ sơ đã lưu.")
            except ValueError as exc:
                st.warning(str(exc))
