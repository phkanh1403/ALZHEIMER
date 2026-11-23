import json
from datetime import date,timedelta
from zipfile import BadZipFile
import pandas as pd
import streamlit as st
from cohort import landmarks
from etl import read_package
from service import snapshot,score,store_data,CONTRACT
from survival import train
import repository as repo

st.set_page_config(page_title="ADNI · Nghiên cứu dọc",layout="wide")
st.title("ADNI · Dự báo lần đầu ghi nhận Dementia")
st.caption("Ứng dụng nghiên cứu cục bộ · ngày gốc là lượt MCI đã quan sát.")
page=st.sidebar.radio("Công việc",["Dữ liệu và đối soát","Mô hình và đánh giá","Hồ sơ và lượt khám","Dự báo","Tổng quan"])
sid=repo.current("data"); mid=repo.current("model")
@st.cache_data(max_entries=4)
def read(kind,key): return repo.get(kind,key)
@st.cache_data(max_entries=3)
def cohort_for(key): return landmarks(read("data",key)["clean"])
data=read("data",sid) if sid else None
def table(records):
    if len(records): st.dataframe(records,use_container_width=True)
def version_picker(kind):
    items=repo.listing(kind); labels={k:f"{d[:19]} · {v} · {k[:10]}" for k,v,d in items}
    if items: return st.selectbox("Phiên bản "+kind,list(labels),format_func=labels.get)
def activate(kind,key):
    if key and st.button("Kích hoạt phiên bản "+kind):
        repo.select(kind,key); st.rerun()
def require_data():
    if data and data["domain"]=="SIM": st.caption("DỮ LIỆU GIẢ LẬP")
    if not data: st.info("Chưa có dataset. Vào Dữ liệu và đối soát."); st.stop()
def select_rid():
    return st.selectbox("Mã hồ sơ",sorted({r["rid"] for r in data["clean"]}))
def local_date(label,value):
    return st.date_input(label,value=value,min_value=date(1900,1,1),max_value=date(2100,12,31))
def files():
    archive=st.file_uploader("ZIP dữ liệu",type="zip"); mapping=st.file_uploader("Mapping JSON đã đối soát",type="json")
    if archive:
        frames=read_package(archive.getvalue())
        table(pd.DataFrame([dict(table=n,rows=len(v),columns=", ".join(v.columns)) for n,v in frames.items()]))
        dd=frames.get("DATADIC")
        if dd is not None and "TBLNAME" in dd:
            with st.expander("Đối chiếu DATADIC của ZIP"):
                key=st.text_input("Tên bảng","DXSUM")
                table(dd[dd.TBLNAME.str.upper()==key.upper()])
        merge=st.checkbox("Bổ sung vào snapshot đang hoạt động",value=False,disabled=not sid)
        if mapping is None:
            from pathlib import Path
            default = Path(__file__).with_name("mapping.dataset.json")
            st.caption("Đang dùng mapping của bản xuất ADNI đã đối chiếu. Với demo, chọn demo_mapping.json.")
        if st.button("Chuẩn hóa và lưu snapshot"):
            snapshot(frames,archive.getvalue(),json.loads(mapping.getvalue() if mapping else default.read_bytes()),sid if merge else None); st.rerun()
    activate("data",version_picker("data"))
    if data:
        st.write("Snapshot đang hoạt động",sid[:12],data["summary"])
        issues=pd.DataFrame(data["issues"]+data["conflicts"])
        if not issues.empty:
            st.caption("Số bản ghi đối soát theo lý do")
            table(issues.reason.value_counts().rename("Số dòng"))
            table(issues.head(500)); st.download_button("Tải toàn bộ dòng đối soát",issues.to_csv(index=False).encode("utf-8-sig"),"doi_soat.csv")
        df=cohort_for(sid)
        table(df.count().rename("Số landmark có giá trị"))
def models():
    require_data()
    st.caption("Chia RID 60/20/20. Các ngưỡng dưới đây là tiêu chí nghiên cứu cấu hình trước khi mở test.")
    events=st.number_input("RID có biến cố tối thiểu",min_value=1,value=5)
    risk=st.number_input("RID còn nguy cơ tối thiểu ở mốc",min_value=2,value=10)
    gap=st.number_input("Giới hạn cho cả khoảng bootstrap 95% của sai lệch hiệu chuẩn trung bình",min_value=0.01,max_value=1.,value=0.10)
    if st.button("Huấn luyện phiên bản ứng viên"):
        with st.spinner("Đang dựng tập và đánh giá…"):
            b=train(cohort_for(sid),42,dict(events=int(events),risk=int(risk),gap=gap))
            b.update(snapshot=sid,contract=CONTRACT,domain=data["domain"])
            k=repo.save("model",f"{len(b['models'])} nhánh · {sid[:10]}",b)
            st.success("Đã lưu ứng viên "+k[:12])
    k=version_picker("model")
    if k:
        b=read("model",k); st.write("Snapshot train",b["snapshot"]); table(pd.DataFrame(b["metrics"]))
        metrics=pd.DataFrame(b["metrics"])
        if not metrics.empty and "brier" in metrics:
            subset=metrics[metrics.split=="test"]
            st.caption("Brier IPCW trên tập test theo ngày")
            if not subset.empty: st.line_chart(subset.pivot_table(index="h",columns="branch",values="brier"))
            for n in subset.branch.unique():
                with st.expander("Ma trận phân loại · "+n):
                    table(subset[subset.branch==n].reindex(columns=["h","threshold","tp","tn","fp","fn","unknown"]))
        st.json({n:dict(mốc=m["supported"],khoảng=m["intervals"]) for n,m in b["models"].items()})
        if b["models"]: activate("model",k)
    st.caption("Mô hình đang hoạt động: "+(mid[:12] if mid else "chưa có"))
def patients():
    require_data()
    if data["clean"]:
        rid=select_rid(); table([r for r in data["clean"] if r["rid"]==rid])
    with st.form("new"):
        rid=st.text_input("RID cho lượt khám (ví dụ LOCAL-001)").strip()
        when=local_date("Ngày khám thực tế",date.today())
        dx=st.selectbox("Chẩn đoán quan sát",["MCI","CN","Dementia"])
        values={}
        for k,label,lo,hi in [("age","Tuổi",18,120),("education","Số năm học",0,40),("mmse","MMSE",0,30),("faq","FAQ",0,30)]:
            values[k]=st.number_input(label,min_value=lo,max_value=hi,value=None)
        sex=st.selectbox("Giới ghi nhận",["Chưa rõ","Nữ","Nam"])
        if sex!="Chưa rõ": values["female"]=int(sex=="Nữ")
        if st.form_submit_button("Lưu lượt khám thành snapshot mới"):
            if not rid or when>date.today(): raise ValueError("Kiểm tra RID và ngày khám thực tế")
            rows=list(data["rows"])
            for kind,v in {"dx":dx,**values}.items():
                if v is not None: rows.append(dict(rid=rid,phase="manual",source="manual",ptid="",kind=kind,date=str(when),value=v,available=str(when),version="",uid=""))
            store_data(rows,data["issues"],data["batches"],sid); st.rerun()
def prediction():
    require_data()
    if not data["clean"]: st.info("Chưa có hồ sơ hợp lệ."); return
    rid=select_rid(); df=cohort_for(sid); eligible=df[df.rid==rid]
    if eligible.empty: st.info("Hồ sơ chưa có lượt MCI đủ điều kiện."); return
    t0=st.selectbox("Lượt khám làm ngày gốc",sorted(eligible.t0,reverse=True))
    target=local_date("Ngày muốn dự báo đến",date.fromisoformat(t0)+timedelta(days=183))
    st.caption(f"Thông tin khóa tại {t0}. Khoảng dự báo {(target-date.fromisoformat(t0)).days} ngày.")
    if not mid: st.info("Chưa kích hoạt mô hình."); return
    bundle=read("model",mid)
    st.write("Chân trời của từng nhánh", {k:m["supported"] for k,m in bundle["models"].items()})
    if st.button("Tính và lưu phiếu"):
        if bundle["domain"]!=data["domain"]: raise ValueError("Mô hình và snapshot khác miền giả lập/ADNI")
        person=[r for r in data["clean"] if r["rid"]==rid]
        p=score(bundle,person,t0,str(target),bool(eligible.loc[eligible.t0==t0,"first"].iloc[0]))
        p.update(model=mid,snapshot=sid,rid=rid)
        repo.save("receipt",rid,p)
        st.success(f"{p['probability']:.2%} · lần đầu ghi nhận Dementia sau {t0} đến hết {target} · "+("nguy cơ nhóm Kaplan–Meier" if p["branch"]=="KM" else p["branch"]))
    for k,label,created in repo.listing("receipt"):
        if label!=rid: continue
        p=read("receipt",k)
        with st.expander(f"{p['t0']} → {p['target']} · {p['probability']:.2%} · {created[:10]}"):
            st.json(p)
            obs=eligible[eligible.t0==p["t0"]]
            if not obs.empty:
                v=obs.iloc[0]; st.write("Đối chiếu với snapshot hiện tại:", "Có biến cố trong hạn" if v.event and v.days<=p["h"] else "Chưa ghi nhận trong hạn, đủ theo dõi" if v.days>=p["h"] else "Chưa đủ theo dõi")
def dashboard():
    require_data(); st.write(data["summary"])
    records=pd.DataFrame(data["clean"])
    if records.empty: return
    latest=records[records.kind=="dx"].sort_values("date").drop_duplicates("rid",keep="last")
    st.subheader("Chẩn đoán ghi nhận gần nhất theo người"); st.bar_chart(latest.value.value_counts())
    for k in ["age","female"]:
        vals=records[records.kind==k].sort_values("date").drop_duplicates("rid",keep="last").value
        if k=="age" and vals.empty:
            birth=records[records.kind=="birthyear"].sort_values("date").drop_duplicates("rid",keep="last")
            joined=latest[["rid","date"]].merge(birth[["rid","value","available"]],on="rid")
            joined=joined[joined.available<=joined.date]
            vals=joined.date.str[:4].astype(int)-joined.value.astype(float)
        st.caption(k+" · số người có giá trị: "+str(len(vals)))
        if len(vals): st.bar_chart(vals.value_counts().sort_index())
try:
    {"Dữ liệu và đối soát":files,"Mô hình và đánh giá":models,"Hồ sơ và lượt khám":patients,"Dự báo":prediction,"Tổng quan":dashboard}[page]()
except (ValueError,KeyError,TypeError,BadZipFile,OSError) as exc:
    st.error(str(exc))
