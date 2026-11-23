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
from registry import initialize, import_zip, LABELS, BOUNDS
initialize(db,ROOT,data)
profiles=pd.read_sql('SELECT * FROM profiles ORDER BY star DESC,name',db)
st.title('Theo dõi MCI')
page=st.sidebar.radio('Chức năng',['Dự đoán','Nhập hồ sơ','Hồ sơ đã lưu','Dashboard mô hình','Dashboard dữ liệu'])
if page=='Nhập hồ sơ':
    manual,batch=st.tabs(['Nhập từng hồ sơ','Nhập ZIP ADNI'])
    with batch:
        upload=st.file_uploader('Thả ZIP chứa DXSUM, MMSE, FAQ, PTDEMOG và UCSFFSX7',type='zip')
        if upload and st.button('Lưu hồ sơ từ ZIP'):
            try:
                added,total=import_zip(db,upload)
                st.success(f'Đã thêm {added}/{total} RID. Các RID đã lưu được bỏ qua.')
            except Exception as exc:
                st.error(f'Không thể nhập ZIP: {exc}')
    with manual:
        st.caption('Nhập tại lần tái khám MCI thứ hai ngoài screening. Có thể để trống phép đo chưa có.')
        with st.form('entry'):
            name=st.text_input('Mã hồ sơ')
            day=st.date_input('Ngày tái khám MCI',min_value=pd.Timestamp('1900-01-01').date())
            eligible=st.checkbox('Đã xác nhận MCI tại mốc và chưa từng ghi nhận Dementia')
            features={}
            for k in cols:
                lo,hi=BOUNDS[k]
                step=1 if isinstance(lo,int) else 0.01
                features[k]=st.number_input(f'{LABELS[k]} · {lo} đến {hi}',min_value=lo,max_value=hi,step=step,value=None)
            st.caption('Giới hạn MRI và tốc độ là rào nhập liệu của ứng dụng. Tốc độ âm thể hiện giảm so với lần đo trước.')
            if st.form_submit_button('Lưu hồ sơ'):
                if not name.strip() or not eligible:st.error('Cần mã hồ sơ và xác nhận điều kiện tại mốc.')
                elif any(features[k] is not None and not lo<=features[k]<=hi for k,lo,hi in [('age',18,115),('education',0,36),('mmse',0,30),('faq',0,30)]) or features['sex'] not in (None,1,2) or any(features[k] is not None and features[k]<=0 for k in ['hippo','entor']):st.error('Kiểm tra phạm vi điểm, tuổi, giới và MRI.')
                else:
                    with db:db.execute('INSERT INTO profiles VALUES(?,?,?,?,0)',(str(uuid.uuid4()),name.strip(),str(day),json.dumps(features)))
                    st.success('Đã lưu. Mở Dự đoán và chọn Hồ sơ đã lưu.')
elif page=='Hồ sơ đã lưu':
    term=st.text_input('Tìm theo tên hoặc RID')
    view=profiles[profiles.name.str.contains(term,case=False,regex=False)]
    if st.checkbox('Chỉ hồ sơ quan trọng'):view=view[view.star==1]
    st.caption(f'{len(view)} hồ sơ · 20 hồ sơ mỗi trang')
    page_number=st.number_input('Trang',1,max(1,(len(view)+19)//20),1)
    for r in view.iloc[(page_number-1)*20:page_number*20].itertuples():
        with st.expander(('★ ' if r.star else '')+r.name+' · '+r.day):
            st.write({LABELS.get(k,k):v for k,v in json.loads(r.features).items() if v is not None})
            if not r.day:st.caption('Chưa có mốc MCI đủ điều kiện dự báo.')
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
        profiles=profiles[profiles.day!='']
        if profiles.empty:st.info('Chưa có hồ sơ đủ mốc dự báo.');st.stop()
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
        st.subheader({'auc_ipcw':'Khả năng phân biệt người có biến cố (AUC)','brier_ipcw':'Sai số xác suất dự báo (Brier)'}[key])
        st.line_chart(metrics.pivot(index='horizon',columns='model',values=key).rename_axis('Thời gian từ tái khám (ngày)'))
        st.caption('AUC càng cao, mô hình càng phân biệt tốt hai nhóm tại mốc đã chọn.' if key=='auc_ipcw' else 'Brier càng thấp, xác suất dự báo càng sát kết cục quan sát sau hiệu chỉnh kiểm duyệt.')
    name=st.selectbox('Mô hình',metrics.model.unique())
    part=metrics[metrics.model==name].set_index('horizon')
    st.subheader('Kết quả phân loại theo thời gian')
    st.bar_chart(part[['tp','tn','fp','fn']].rename(columns=dict(tp='Dương đúng',tn='Âm đúng',fp='Dương sai',fn='Âm sai')).rename_axis('Thời gian (ngày)'))
    st.caption('Số người theo bốn kết quả phân loại tại ngưỡng nguy cơ 50%. Người bị kiểm duyệt sớm được thống kê riêng.')
    st.dataframe(part.rename(columns=dict(n="Số người",cases="Có biến cố",controls="Theo dõi qua mốc",censored_early="Kiểm duyệt sớm",mean_prediction="Nguy cơ trung bình",observed_ipcw="Tỷ lệ quan sát IPCW",auc_ipcw="AUC",brier_ipcw="Brier",tp="Dương đúng",tn="Âm đúng",fp="Dương sai",fn="Âm sai")).rename_axis("Ngày"))
    st.write('Tương quan Spearman trên train')
    st.dataframe(data[MODELS[name]].rename(columns=LABELS).corr(method='spearman'))
    st.caption('Hệ số từ −1 đến 1 biểu thị chiều và mức độ liên hệ thứ hạng giữa hai biến.')
else:
    st.header('Tổng quan hồ sơ đã lưu')
    sources=profiles.id.map(lambda k:'ADNI train' if k.startswith('Train:') else ('Nhập ZIP' if k.startswith('ZIP:') else 'Nhập tay'))
    st.bar_chart(sources.value_counts().rename('Số hồ sơ'))
    st.caption('Số hồ sơ đang lưu theo nguồn, bao gồm cả hồ sơ chưa đủ mốc dự báo.')
    dx=read(ROOT/'data/train','DXSUM').dropna(subset=['EXAMDATE'])
    latest=dx[dx.DIAGNOSIS.isin([1,2,3])].sort_values('EXAMDATE').drop_duplicates('RID',keep='last')
    for title,series in [('Chẩn đoán gần nhất trong train',latest.DIAGNOSIS.map({1:'CN',2:'MCI',3:'Dementia'})),('Giới tại mốc MCI',data.sex.map({1:'Nam',2:'Nữ'})),('Tuổi tại mốc MCI',pd.cut(data.age,bins=range(40,111,5)).astype(str)),('Điểm MMSE tại mốc',data.mmse)]:
        st.subheader(title)
        st.bar_chart(series.dropna().value_counts().sort_index().rename('Số người'))
        st.caption('Mỗi RID có chẩn đoán hợp lệ được đếm một lần theo lượt gần nhất.' if title.startswith('Chẩn') else 'Phân bố trong nhóm train tại lần tái khám MCI được chọn làm mốc dự báo.')

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
