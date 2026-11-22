import pandas as pd
import streamlit as st
from storage import history
st.subheader("Hồ sơ trong hệ thống")
records=history()
rows=[dict(r["payload"],patient=r["patient"]) for r in records if r["kind"]=="visit"]
if rows:
    df=pd.DataFrame(rows).drop_duplicates(["patient","date"],keep="last")
    latest=df.sort_values("date").drop_duplicates("patient",keep="last")
    st.metric("Số người",len(latest))
    st.dataframe(latest)
    st.bar_chart(latest.dx.value_counts())
    st.bar_chart(latest.female.map({0:"Nam",1:"Nữ"}).fillna("Chưa rõ").value_counts())
    st.bar_chart(pd.cut(latest.age.astype(float),bins=range(15,126,5)).value_counts(sort=False).rename_axis("Tuổi").rename(index=str))
    pred=[dict(r["payload"],patient=r["patient"]) for r in records if r["kind"]=="prediction"]
    if pred:
        st.dataframe(pd.DataFrame(pred)[["patient","t0","model","probability"]])
else:
    st.info("Tạo hồ sơ đầu tiên ở trang nhập thủ công.")
