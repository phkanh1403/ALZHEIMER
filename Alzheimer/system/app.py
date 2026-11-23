import json
import sqlite3
import uuid
import numpy as np
import pandas as pd
import streamlit as st
from pipeline import ROOT, MODELS, predict, read

st.set_page_config(page_title='ADNI',layout='wide')
runs=ROOT/'runs'
models=json.loads((runs/'models.json').read_text())
data=pd.read_csv(runs/'train_landmarks.csv')
metrics=pd.read_csv(runs/'evaluation.csv')
cols=MODELS['M3']
db=sqlite3.connect(ROOT/'profiles.sqlite')
db.execute('CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0)')
profiles=pd.read_sql('SELECT * FROM profiles ORDER BY star DESC,name',db)
st.title('Theo dõi MCI')
page=st.sidebar.radio('Chức năng',['Dự đoán','Nhập hồ sơ','Hồ sơ đã lưu','Dashboard mô hình','Dashboard dữ liệu'])
if page=='Nhập hồ sơ':
    st.caption('Nhập tại lần tái khám MCI thứ hai ngoài screening. Có thể để trống phép đo chưa có.')
    with st.form('entry'):
        name=st.text_input('Mã hồ sơ')
        day=st.date_input('Ngày tái khám MCI',min_value=pd.Timestamp('1900-01-01').date())
        eligible=st.checkbox('Đã xác nhận MCI tại mốc và chưa từng ghi nhận Dementia')
        labels=['Tuổi (năm)','Giới ADNI (1 nam, 2 nữ)','Học vấn (năm)','MMSE','FAQ','Tổng hồi hải mã hai bên (mm³)','Độ dày nội khứu trung bình (mm)','Thay đổi MMSE (điểm/năm)','Thay đổi FAQ (điểm/năm)','Thay đổi hồi hải mã (mm³/năm)','Thay đổi nội khứu (mm/năm)']
        features={k:st.number_input(label,value=None) for k,label in zip(cols,labels)}
        if st.form_submit_button('Lưu hồ sơ'):
            if not name.strip() or not eligible:st.error('Cần mã hồ sơ và xác nhận điều kiện tại mốc.')
            elif any(features[k] is not None and not lo<=features[k]<=hi for k,lo,hi in [('age',18,115),('education',0,36),('mmse',0,30),('faq',0,30)]) or features['sex'] not in (None,1,2) or any(features[k] is not None and features[k]<=0 for k in ['hippo','entor']):st.error('Kiểm tra phạm vi điểm, tuổi, giới và MRI.')
            else:
                with db:db.execute('INSERT INTO profiles VALUES(?,?,?,?,0)',(str(uuid.uuid4()),name.strip(),str(day),json.dumps(features)))
                st.success('Đã lưu. Mở Dự đoán và chọn Hồ sơ đã lưu.')
elif page=='Hồ sơ đã lưu':
    for r in profiles.itertuples():
        with st.expander(('★ ' if r.star else '')+r.name+' · '+r.day):
            st.write(json.loads(r.features))
            if st.button('Bỏ đánh dấu' if r.star else 'Đánh dấu quan trọng',key='s'+r.id):
                with db:db.execute('UPDATE profiles SET star=1-star WHERE id=?',(r.id,))
                st.rerun()
            confirm=st.checkbox('Xác nhận xóa',key='c'+r.id)
            if st.button('Xóa',key='d'+r.id,disabled=not confirm):
                with db:db.execute('DELETE FROM profiles WHERE id=?',(r.id,))
                st.rerun()
    if profiles.empty:st.info('Thêm hồ sơ tại mục Nhập hồ sơ.')
elif page=='Dự đoán':
    src=st.radio('Nguồn',['ADNI train','Hồ sơ đã lưu'])
    if src=='ADNI train':
        complete=st.checkbox('Chỉ hiện RID đủ đầu vào cho cả bốn mô hình',value=True)
        subset=data.dropna(subset=cols) if complete else data
        if subset.empty:st.info('Chưa có RID đủ bốn bộ biến.');st.stop()
        rid=st.selectbox('RID',subset.RID)
        row=subset[subset.RID==rid]
        day=row.t0.iloc[0]
    else:
        if profiles.empty:st.info('Thêm hồ sơ tại mục Nhập hồ sơ.');st.stop()
        ix=st.selectbox('Hồ sơ',profiles.index,format_func=lambda i:profiles.loc[i,'name']+' · '+profiles.loc[i,'day'])
        day=profiles.loc[ix,'day']
        row=pd.DataFrame([json.loads(profiles.loc[ix,'features'])])
    h=st.selectbox('Chân trời khảo sát (ngày)',[183,365,730])
    st.write('Ngày tái khám',day,' · Ngày đích',(pd.Timestamp(day)+pd.Timedelta(days=h)).date())
    if st.button('Dự đoán'):
        result=[]
        for name,entry in models.items():
            model=entry['model']
            missing=row[MODELS[name]].columns[row[MODELS[name]].isna().any()].tolist()
            risk='Thiếu '+', '.join(missing) if missing else 'Chưa đủ dữ liệu huấn luyện'
            if model and not missing and h<=max(model['times']):risk=f'{predict(model,row,h)[0]:.2%}'
            interval='—'
            if model and not missing and (runs/'bootstrap.json').exists():
                draws=json.loads((runs/'bootstrap.json').read_text())[name]
                low,high=np.quantile([predict(m,row,h)[0] for m in draws],[.025,.975])
                interval=f'{low:.2%} – {high:.2%}'
            result.append({'model':name,'Nguy cơ ghi nhận Dementia':risk,'Khoảng bootstrap 95%':interval})
        st.dataframe(pd.DataFrame(result).merge(metrics[metrics.horizon==h],on='model',how='left'))
        st.caption('Khoảng bootstrap phản ánh biến thiên do lấy mẫu train. AUC và Brier đo hiệu năng trên tập test ở ngày đã chọn. Chưa xác lập độ tin cậy cá nhân hoặc ngày tối đa đáng tin cậy.')
elif page=='Dashboard mô hình':
    st.write('Đánh giá trên test theo từng chân trời')
    if st.checkbox('So sánh trên cùng RID test đủ cả bốn bộ biến',value=True):
        metrics=pd.read_csv(runs/'paired.csv')
    for key in ['auc_ipcw','brier_ipcw']:
        st.subheader(key)
        st.line_chart(metrics.pivot(index='horizon',columns='model',values=key))
    name=st.selectbox('Mô hình',metrics.model.unique())
    part=metrics[metrics.model==name].set_index('horizon')
    st.bar_chart(part[['tp','tn','fp','fn']])
    st.dataframe(part)
    st.write('Tương quan Spearman trên train')
    st.dataframe(data[MODELS[name]].corr(method='spearman'))
else:
    dx=read(ROOT/'data/train','DXSUM').dropna(subset=['EXAMDATE'])
    latest=dx[dx.DIAGNOSIS.isin([1,2,3])].sort_values('EXAMDATE').drop_duplicates('RID',keep='last')
    for title,series in [('Chẩn đoán gần nhất trong train',latest.DIAGNOSIS.map({1:'CN',2:'MCI',3:'Dementia'})),('Giới tại mốc MCI',data.sex.map({1:'Nam',2:'Nữ'})),('Tuổi tại mốc MCI',pd.cut(data.age,bins=range(40,111,5)).astype(str)),('Điểm MMSE tại mốc',data.mmse)]:
        st.subheader(title)
        st.bar_chart(series.dropna().value_counts().sort_index())

if page=="Dashboard mô hình":

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
