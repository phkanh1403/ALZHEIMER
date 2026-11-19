import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
from cohort import build, SETS
from train import probability

st.title("Dự báo MCI tại tái khám")
st.caption("Trình diễn bằng hồ sơ giả lập · chân trời 183 ngày")
folder = Path("runs/demo")
if not (folder/"models.joblib").is_file():
    st.info("Chạy python train.py demo runs/demo trước khi mở website.")
    st.stop()
data,_ = build("demo")
results = pd.read_csv(folder/"predictions.csv")
rid = st.selectbox("Hồ sơ giả lập thuộc tập test",results.rid)
row = data[data.rid.eq(rid)]
tabs = st.tabs(["Hồ sơ","Dự báo","Dashboard"])
with tabs[0]:
    history = pd.read_csv("demo/measures.csv")
    st.dataframe(history[history.rid.eq(rid)].sort_values("date"))
with tabs[1]:
    st.write("Ngày tái khám",row.iloc[0].t0.date())
    models = joblib.load(folder/"models.joblib")
    for name in ("M2","M3"):
        st.metric(name,f"{probability(models[name],row[SETS[name]])[0]:.1%}")
with tabs[2]:
    st.caption("Chỉ số trên dữ liệu giả lập")
    st.dataframe(pd.read_csv(folder/"metrics.csv"))
