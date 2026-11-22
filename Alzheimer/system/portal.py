from pathlib import Path
import runpy
import streamlit as st
import pandas as pd
from sqlalchemy import delete
from storage import engine,entries,record,history
from workflow import folder,select_patient
from cohort import build
from dashboards import model_dashboard
import zip_import

page=st.sidebar.radio("Điều hướng",["Nhập hồ sơ mới","Dự đoán","Hồ sơ đã lưu","Dashboard"])
st.title(page)
st.caption(f"Bộ mô hình: {folder}" + (" · huấn luyện trên dữ liệu mô phỏng" if folder==Path("runs/demo") else ""))
if page=="Nhập hồ sơ mới":
    manual,archive=st.tabs(["Nhập thủ công một hồ sơ","Nhập file ZIP"])
    with manual:
        runpy.run_module("intake")
    with archive:
        zip_import.main()
elif page=="Dự đoán":
    runpy.run_module("predict_view")
elif page=="Hồ sơ đã lưu":
    patient=select_patient()
    if patient:
        items=history(patient)
        stars=[r["payload"]["important"] for r in items if r["kind"]=="flag"]
        important=bool(stars and stars[-1])
        if st.button("★ Bỏ đánh dấu quan trọng" if important else "☆ Đánh dấu quan trọng"):
            record(patient,"flag",dict(important=not important))
            st.rerun()
        st.dataframe([dict(r["payload"],loai=r["kind"]) for r in items if r["kind"]!="flag"])
        with st.expander("Xóa hồ sơ"):
            confirm=st.checkbox("Xác nhận xóa toàn bộ lượt khám và phiếu dự báo",key="confirm_"+patient)
            if st.button("Xóa hồ sơ này",disabled=not confirm):
                with engine.begin() as conn:
                    conn.execute(delete(entries).where(entries.c.patient==patient))
                st.rerun()
    else:
        st.info("Chưa có hồ sơ.")
else:
    model,system=st.tabs(["Dashboard mô hình","Dashboard hệ thống"])
    with model:
        if (folder/"metrics.csv").exists():
            model_dashboard(pd.read_csv(folder/"predictions.csv"),pd.read_csv(folder/"cohort.csv"),pd.read_csv(folder/"metrics.csv"))
        else:
            st.info("Chưa có kết quả đánh giá.")
    with system:
        runpy.run_module("system_chart")
