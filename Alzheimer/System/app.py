from datetime import date, timedelta
import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
from cohort import build, SETS
from train import probability, H
from dashboards import model_dashboard, profile_dashboard
from manual import make_features, missing_features

st.title("Dự báo MCI tại tái khám")
st.caption("Trình diễn nghiên cứu · mô hình huấn luyện trên dữ liệu giả lập · 183 ngày")
folder = Path("runs/demo")
if not (folder/"models.joblib").is_file():
    st.info("Chạy python train.py demo runs/demo trước khi mở website.")
    st.stop()
data,_ = st.cache_data(build)("demo")
results = pd.read_csv(folder/"predictions.csv")
rid = st.selectbox("Hồ sơ giả lập thuộc tập test",results.rid)
row = data[data.rid.eq(rid)]
tabs = st.tabs(["Hồ sơ mẫu","Dự báo hồ sơ mẫu","Dashboard mô hình","Nhập hồ sơ mới","Dashboard hồ sơ"])
with tabs[0]:
    history = pd.read_csv("demo/measures.csv")
    st.dataframe(history[history.rid.eq(rid)].sort_values("date"))
models = joblib.load(folder/"models.joblib")
with tabs[1]:
    st.write("Ngày tái khám",row.iloc[0].t0.date())
    for name in ("M2","M3"):
        st.metric(name,f"{probability(models[name],row[SETS[name]])[0]:.1%}")
with tabs[2]:
    model_dashboard(results, data, pd.read_csv(folder/"metrics.csv"))
with tabs[4]:
    profile_dashboard(pd.read_csv("demo/visits.csv", parse_dates=["date"]), results)

with tabs[3]:
    st.subheader("Dự báo tại lần tái khám")
    st.caption("Dùng số liệu giả lập hoặc đã khử định danh để trình diễn.")
    with st.form("new_profile"):
        eligible = st.checkbox("Hồ sơ có MCI ở hai mốc và chưa từng được ghi nhận Dementia đến lần tái khám", key="manual_mci_eligible")
        age = st.number_input("Tuổi tại tái khám", min_value=18, max_value=120, value=72)
        sex = st.selectbox("Giới", ["Nữ", "Nam"])
        education = st.number_input("Số năm học", min_value=0, max_value=40, value=16)
        visits_data = []
        imaging = []
        for i, (label, default) in enumerate([
            ("Mốc MCI đầu", date(2024, 1, 12)),
            ("Tái khám", date(2024, 7, 12)),
        ]):
            st.markdown(f"**{label}**")
            anchor = st.date_input("Ngày khám", default, key=f"visit_{i}")
            mmse = st.number_input("MMSE", 0, 30, 27 - 2*i, key=f"mmse_{i}")
            faq = st.number_input("FAQ", 0, 30, 3 + 3*i, key=f"faq_{i}")
            visits_data.append((anchor, mmse, faq))
            has_mri = st.checkbox("Có số đo MRI ở mốc này", value=True, key=f"has_mri_{i}")
            scan = st.date_input("Ngày chụp MRI", default, key=f"scan_{i}")
            hippo = st.number_input("Tổng thể tích hồi hải mã hai bên (mm³)",
                                    min_value=1.0, value=6200.0-200*i, key=f"hippo_{i}")
            entor = st.number_input("Độ dày nội khứu trung bình hai bên (mm)",
                                    min_value=0.01, value=2.75-0.1*i, key=f"entor_{i}")
            qc = st.checkbox("Số đo MRI đạt QC", value=True, key=f"qc_{i}")
            version = st.text_input("Phiên bản xử lý MRI", "demo-v1", key=f"version_{i}")
            if has_mri:
                imaging.append((anchor, scan, hippo, entor, qc, version))
        submitted = st.form_submit_button("Tính dự báo")
    if submitted:
        try:
            if not eligible:
                raise ValueError("Cần xác nhận điều kiện chẩn đoán trước khi tính dự báo.")
            first, followup = (visit[0] for visit in visits_data)
            features = make_features(age, int(sex == "Nữ"), education, first, followup,
                                     visits_data, imaging)
            missing = missing_features(features)
            st.write(f"Mốc dự báo {followup:%d/%m/%Y} · Hạn dự báo {followup + timedelta(days=H):%d/%m/%Y}")
            for name in ("M2", "M3"):
                if missing[name]:
                    st.warning(f"{name} thiếu phép đo hợp lệ để tính dự báo.")
                elif missing["M3"]:
                    st.info(f"{name} chưa được đánh giá cho nhóm hồ sơ thiếu cặp MRI hợp lệ.")
                else:
                    risk = probability(models[name], features[SETS[name]])[0]
                    st.metric(f"{name} · Xác suất ghi nhận Dementia trong 183 ngày", f"{risk:.1%}")
            with st.expander("Đặc trưng đã tính"):
                st.dataframe(features[SETS["M3"]], hide_index=True)
        except ValueError as error:
            st.error(str(error))
