import json
import streamlit as st
from pipeline import ROOT, MODELS, pd, predict, read

st.set_page_config(page_title='ADNI · Dự báo MCI',layout='wide')
st.title('Theo dõi MCI trong dữ liệu ADNI')
runs=ROOT/'runs'
if not (runs/'models.json').exists():
    st.info('Chạy python train.py để tạo mô hình từ dữ liệu train đi kèm.')
    st.stop()
models=json.loads((runs/'models.json').read_text())
data=pd.read_csv(runs/'train_landmarks.csv')
overview,patient,quality=st.tabs(['Đoàn hệ train','Dự báo nghiên cứu','Đánh giá độc lập'])
with overview:
    st.metric('Người có mốc tái khám MCI',len(data))
    dx=read(ROOT/'data/train','DXSUM').dropna(subset=['EXAMDATE'])
    dx=dx[dx.DIAGNOSIS.isin([1,2,3])].sort_values('EXAMDATE')
    dx=dx.groupby('RID').tail(1)
    st.caption('Chẩn đoán có mã hợp lệ gần nhất trong DXSUM của train')
    st.bar_chart(dx.DIAGNOSIS.map({1:'CN',2:'MCI',3:'Dementia'}).value_counts())
    st.dataframe(data[['age','sex','education']].describe())
with patient:
    rid=st.selectbox('RID có mốc hợp lệ',data.RID)
    row=data[data.RID==rid]
    anchor=pd.Timestamp(row.t0.iloc[0])
    st.write('Ngày tái khám MCI làm mốc',anchor.date())
    name=st.selectbox('Mô hình',MODELS)
    entry=models[name]
    st.dataframe(row.reindex(columns=MODELS[name]).T.rename(columns={row.index[0]:'Giá trị tại mốc'}))
    if not entry['model']:st.info('Chưa đủ người hoặc biến cố để khớp mô hình.')
    elif row[MODELS[name]].isna().any(axis=None):st.info('Hồ sơ thiếu đặc trưng của mô hình đã chọn.')
    else:
        h=st.selectbox('Khoảng dự báo khảo sát (ngày)',[183,365,730])
        st.write('Ngày đích',(anchor+pd.Timedelta(days=h)).date())
        if h<=entry['model']['times'][-1]:
            st.metric('Ước lượng nghiên cứu trên hồ sơ train',f"{predict(entry['model'],row,h)[0]:.1%}")
            st.caption('Chân trời khảo sát chưa được xác nhận cho sử dụng lâm sàng.')
        else:st.info('Ngoài thời gian có biến cố quan sát trong train.')
with quality:
    st.dataframe(pd.DataFrame([{k:v for k,v in e.items() if k!='model'}|{'model':n} for n,e in models.items()]))
    if (runs/'evaluation.csv').exists():st.dataframe(pd.read_csv(runs/'evaluation.csv'))
    else:st.info('Giải nén ZIP test và chạy evaluate.py để đánh giá.')
with quality:
    import tempfile
    import zipfile
    from evaluate import evaluate
    upload = st.file_uploader('Chọn ADNI_Test_20.zip để tính lại chỉ số',type='zip')
    if upload and st.button('Đánh giá bộ test'):
        try:
            with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(upload) as archive:
                allowed={'DXSUM','PTDEMOG','MMSE','FAQ','UCSFFSX7'}
                from pathlib import Path
                import shutil
                for name in allowed:
                    items=[i for i in archive.infolist() if Path(i.filename).name==name+'.csv']
                    if len(items)!=1 or items[0].file_size>200_000_000:
                        raise ValueError(f'Tệp {name} thiếu, trùng hoặc vượt giới hạn kích thước.')
                    with archive.open(items[0]) as src, (Path(folder)/(name+'.csv')).open('wb') as dst:
                        shutil.copyfileobj(src,dst)
                with st.spinner('Đang dựng mốc và đánh giá'):
                    st.dataframe(evaluate(folder))
        except (ValueError,OSError,zipfile.BadZipFile,KeyError) as err:
            st.error(str(err))
