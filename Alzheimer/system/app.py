"""Ứng dụng Streamlit v2: dự báo nguy cơ sa sút trí tuệ tại một lần tái khám MCI."""
import json
import sqlite3
import tempfile
import uuid
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from pipeline import ROOT, MODELS, LABELS, HORIZONS, PRIMARY, MIN_SPAN, blup_slope, read
from model import predict
from registry import initialize, import_zip, BOUNDS, FIELDS

st.set_page_config(page_title='ADNI MCI v2', layout='wide')
runs = ROOT / 'runs'
prep = json.loads((runs / 'preprocessing.json').read_text())
secondary = json.loads((runs / 'secondary_models.json').read_text())
data = pd.read_csv(runs / 'train_landmarks.csv', parse_dates=['t0'])


@st.cache_data
def boot():
    p = runs / 'bootstrap.json'
    return json.loads(p.read_text()) if p.exists() else {}


def table(name):
    p = runs / name
    return pd.read_csv(p) if p.exists() else None


db = sqlite3.connect(ROOT / 'profiles.sqlite', check_same_thread=False)
db.execute('CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0)')
initialize(db, data)
profiles = pd.read_sql('SELECT * FROM profiles ORDER BY star DESC,name', db)

st.title('Theo dõi MCI · v2')
st.caption('Phạm vi sử dụng: người MCI CHƯA dùng kháng thể kháng amyloid, có ≥ 2 MRI T1 cùng máy/cường độ từ trường '
           f'cách nhau ≥ {MIN_SPAN} ngày. Công cụ hỗ trợ phân tầng theo dõi, không dùng để chẩn đoán hay quyết định điều trị.')
page = st.sidebar.radio('Chức năng', ['Dự đoán', 'Nhập hồ sơ', 'Hồ sơ đã lưu', 'Dashboard mô hình', 'Dashboard dữ liệu'])


def risk_table(row, h):
    result = []
    draws = boot()
    for name in MODELS:
        model = secondary[name]
        missing = [LABELS[c] for c in MODELS[name] if pd.isna(row.get(c))]
        if missing:
            result.append({'Mô hình': name, f'Nguy cơ đến {h} ngày': 'Thiếu: ' + ', '.join(missing), 'Khoảng bootstrap 95%': '—'})
            continue
        frame = pd.DataFrame([{c: row[c] for c in MODELS[name]}])
        p = predict(model, frame, h)[0]
        interval = '—'
        if name in draws:
            q = np.quantile([predict(m, frame, h)[0] for m in draws[name]], [.025, .975])
            interval = f'{q[0]:.1%} – {q[1]:.1%}'
        result.append({'Mô hình': name, f'Nguy cơ đến {h} ngày': f'{p:.1%}', 'Khoảng bootstrap 95%': interval})
    return pd.DataFrame(result)


if page == 'Dự đoán':
    src = st.radio('Nguồn', ['ADNI train', 'Hồ sơ đã lưu'])
    if src == 'ADNI train':
        complete = st.checkbox('Chỉ hiện mốc đủ đầu vào cho cả bốn mô hình', value=True)
        subset = data.dropna(subset=MODELS['M3']) if complete else data
        rid = st.selectbox('RID', sorted(subset.RID.unique()))
        lm = subset[subset.RID == rid].sort_values('t0')
        when = st.selectbox('Mốc (lần tái khám MCI)', lm.t0.dt.date.astype(str).tolist(), index=len(lm) - 1)
        row = lm[lm.t0.dt.date.astype(str) == when].iloc[0].to_dict()
        day = when
    else:
        usable = profiles[profiles.day != '']
        if usable.empty:
            st.info('Chưa có hồ sơ đủ mốc dự báo.')
            st.stop()
        ix = st.selectbox('Hồ sơ', usable.index, format_func=lambda i: usable.loc[i, 'name'] + ' · ' + usable.loc[i, 'day'])
        day = usable.loc[ix, 'day']
        row = json.loads(usable.loc[ix, 'features'])
    h = st.selectbox('Chân trời (ngày)', HORIZONS, index=HORIZONS.index(PRIMARY))
    st.write('Ngày mốc', day, '· Ngày đích', (pd.Timestamp(day) + pd.Timedelta(days=h)).date())
    treated = st.checkbox('Bệnh nhân đang dùng thuốc kháng amyloid (lecanemab, donanemab…)')
    span = row.get('mri_span_days')
    if span is not None and not pd.isna(span) and span < MIN_SPAN:
        st.warning(f'Hai MRI cùng máy chỉ cách nhau {int(span)} ngày (< {MIN_SPAN}): tốc độ teo không đáng tin, M3 không áp dụng.')
    if treated:
        st.error('Ngoài phạm vi sử dụng: điều trị kháng amyloid gây giả teo (ARPA) làm sai lệch tốc độ teo. Không hiển thị dự báo.')
    elif st.button('Dự đoán'):
        st.dataframe(risk_table(row, h), hide_index=True)
        st.caption('Xác suất ghi nhận Dementia đến ngày đích, với giả định người bệnh còn sống. Khoảng bootstrap phản ánh '
                   'biến thiên do lấy mẫu train (200 lần, theo cụm người), không phải độ tin cậy cá nhân đã được kiểm chứng.')
        d = table('validation_deltas.csv')
        if d is not None:
            st.caption('Giá trị gia tăng của MRI lặp (M3 − M2) trong kiểm định chéo: xem Dashboard mô hình.')

elif page == 'Nhập hồ sơ':
    manual, batch = st.tabs(['Nhập từng hồ sơ', 'Nhập ZIP ADNI'])
    with batch:
        up = st.file_uploader('ZIP chứa DXSUM, MMSE, FAQ, ADAS, NEUROBAT, PTDEMOG, UCSFFSX7 (FOXLABBSI tùy chọn)', type='zip')
        if up and st.button('Lưu hồ sơ từ ZIP'):
            try:
                added, total, overlap = import_zip(db, up, prep, set(data.RID))
                st.success(f'Đã thêm {added} hồ sơ từ {total} RID. {overlap} RID trùng train bị bỏ qua.')
            except Exception as exc:
                st.error(f'Không thể nhập ZIP: {exc}')
    with manual:
        st.caption('Nhập tại một lần tái khám MCI. Có thể để trống phép đo chưa có.')
        with st.expander('Tính slope từ chuỗi đo (mô hình hỗn hợp, tham số khóa từ train)'):
            st.caption('Mỗi dòng: YYYY-MM-DD,giá trị. Hồi hải mã: nhập thể tích đã hiệu chỉnh ICV. '
                       f'Hiệu chỉnh: HV_adj = HV − {prep["icv"]["b"]:.5f} × (ICV − {prep["icv"]["icv_mean"]:.0f}).')
            kind = st.selectbox('Đại lượng', ['hippo_adj', 'entor', 'adas13', 'faq'], format_func=lambda k: LABELS.get(k, k))
            text = st.text_area('Chuỗi đo', '2023-01-10,6500\n2024-01-15,6380')
            if st.button('Tính slope'):
                try:
                    pts = pd.DataFrame([l.split(',') for l in text.strip().splitlines()], columns=['d', 'v'])
                    pts['d'] = pd.to_datetime(pts.d)
                    pts = pts.sort_values('d')
                    t = ((pts.d - pts.d.iloc[0]).dt.days / 365.25).to_numpy()
                    if t[-1] * 365.25 < MIN_SPAN:
                        st.warning(f'Chuỗi trải dưới {MIN_SPAN} ngày: không tính slope.')
                    else:
                        st.success(f'Slope (BLUP): {blup_slope(t, pts.v.astype(float).to_numpy(), prep[kind]):.3f} đơn vị/năm')
                except Exception as exc:
                    st.error(f'Không đọc được chuỗi: {exc}')
        with st.form('entry'):
            name = st.text_input('Mã hồ sơ')
            day = st.date_input('Ngày tái khám MCI', min_value=pd.Timestamp('1900-01-01').date())
            eligible = st.checkbox('Đã xác nhận MCI tại mốc, chưa từng ghi nhận Dementia, chưa dùng thuốc kháng amyloid')
            feats = {}
            for k in FIELDS:
                if k == 'tau2':
                    continue
                lo, hi = BOUNDS[k]
                feats[k] = st.number_input(f'{LABELS[k]} · {lo:g} đến {hi:g}', min_value=lo, max_value=hi, value=None)
            span = st.number_input('Khoảng cách MRI đầu–cuối cùng máy (ngày)', min_value=0, max_value=10000, value=None)
            if st.form_submit_button('Lưu hồ sơ'):
                if not name.strip() or not eligible:
                    st.error('Cần mã hồ sơ và xác nhận điều kiện tại mốc.')
                else:
                    feats['tau2'] = None if feats['tau'] is None else feats['tau'] ** 2
                    feats['mri_span_days'] = span
                    with db:
                        db.execute('INSERT INTO profiles VALUES(?,?,?,?,0)', (str(uuid.uuid4()), name.strip(), str(day), json.dumps(feats)))
                    st.success('Đã lưu. Mở Dự đoán và chọn Hồ sơ đã lưu.')

elif page == 'Hồ sơ đã lưu':
    term = st.text_input('Tìm theo tên hoặc RID')
    view = profiles[profiles.name.str.contains(term, case=False, regex=False)]
    if st.checkbox('Chỉ hồ sơ quan trọng'):
        view = view[view.star == 1]
    st.caption(f'{len(view)} hồ sơ · 20 hồ sơ mỗi trang')
    pg = st.number_input('Trang', 1, max(1, (len(view) + 19) // 20), 1)
    for r in view.iloc[(pg - 1) * 20:pg * 20].itertuples():
        with st.expander(('★ ' if r.star else '') + r.name + ' · ' + r.day):
            st.write({LABELS.get(k, k): v for k, v in json.loads(r.features).items() if v is not None})
            if st.button('Bỏ đánh dấu' if r.star else 'Đánh dấu quan trọng', key='s' + r.id):
                with db:
                    db.execute('UPDATE profiles SET star=1-star WHERE id=?', (r.id,))
                st.rerun()
            ok = st.checkbox('Xác nhận xóa', key='c' + r.id)
            if st.button('Xóa', key='d' + r.id, disabled=not ok):
                with db:
                    db.execute('DELETE FROM profiles WHERE id=?', (r.id,))
                st.rerun()

elif page == 'Dashboard mô hình':
    st.subheader('Độ bao phủ (train)')
    st.dataframe(table('coverage.csv'), hide_index=True)
    val = table('validation.csv')
    if val is not None:
        st.subheader('Kiểm định chéo lặp 5 fold × 20 lần (train, một mốc ngẫu nhiên/người)')
        v = val[val.scope == 'random']
        st.line_chart(v.pivot(index='h', columns='model', values='auc').rename_axis('Chân trời (ngày)'))
        st.caption('td-AUC có trọng số IPCW theo chân trời.')
        st.dataframe(v[['h', 'model', 'n', 'cases', 'auc', 'ipa', 'oe_km', 'cal_slope_km', 'nb_core']].round(4), hide_index=True)
        st.caption('IPA = 1 − Brier/Brier mô hình null. O/E và slope hiệu chuẩn tính bằng Kaplan–Meier theo 5 nhóm nguy cơ. '
                   'nb_core = net benefit trung bình ở ngưỡng 15–30%.')
    d = table('validation_deltas.csv')
    if d is not None:
        st.subheader('Chênh lệch ghép cặp và kết luận theo ngưỡng δ = 0,02 (định trước)')
        st.dataframe(d.round(4), hide_index=True)
    s = table('sensitivity.csv')
    if s is not None:
        st.subheader('Phân tích độ nhạy')
        st.dataframe(s.round(4), hide_index=True)
    tm = table('test_metrics.csv')
    if tm is not None:
        st.subheader('Tập test 20% (mô hình khóa từ train)')
        st.dataframe(tm[tm.scope == 'random'][['h', 'model', 'n', 'cases', 'auc', 'ipa', 'oe_km', 'cal_slope_km', 'nb_core']].round(4), hide_index=True)
        st.dataframe(table('test_deltas.csv').round(4), hide_index=True)
        cal = table('test_calibration.csv')
        hh = st.selectbox('Hiệu chuẩn test tại chân trời', HORIZONS, index=1)
        c = cal[(cal.h == hh) & cal.model.isin(['M2', 'M3'])]
        st.scatter_chart(c, x='predicted', y='observed_km', color='model')
        st.caption('Mỗi điểm là một nhóm nguy cơ (5 nhóm): nguy cơ dự báo trung bình so với Kaplan–Meier quan sát.')
    up = st.file_uploader('Chấm lại bằng ZIP test (thư mục test đã nén)', type='zip')
    if up and st.button('Đánh giá bộ test'):
        from evaluate import evaluate
        try:
            with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(up) as archive:
                for i in archive.infolist():
                    if i.filename.endswith('.csv') and Path(i.filename).name.split('.')[0] in \
                            {'DXSUM', 'PTDEMOG', 'MMSE', 'FAQ', 'ADAS', 'NEUROBAT', 'UCSFFSX7', 'FOXLABBSI'}:
                        archive.extract(i, folder)
                with st.spinner('Đang dựng mốc và đánh giá (khoảng 1 phút)'):
                    m, dl, info = evaluate(folder)
                st.write(info)
                st.dataframe(dl.round(4), hide_index=True)
        except Exception as err:
            st.error(str(err))

else:
    st.header('Tổng quan dữ liệu train')
    src = profiles.id.map(lambda k: 'ADNI train' if k.startswith('Train:') else ('Nhập ZIP' if k.startswith('ZIP:') else 'Nhập tay'))
    st.bar_chart(src.value_counts().rename('Số hồ sơ'))
    dx = read(ROOT / 'data' / 'train', 'DXSUM').dropna(subset=['EXAMDATE'])
    latest = dx[dx.DIAGNOSIS.isin([1, 2, 3])].sort_values('EXAMDATE').drop_duplicates('RID', keep='last')
    one = data.sort_values('t0').drop_duplicates('RID')
    for title, series in [('Chẩn đoán gần nhất', latest.DIAGNOSIS.map({1: 'CN', 2: 'MCI', 3: 'Dementia'})),
                          ('Số mốc landmark mỗi người', data.groupby('RID').size()),
                          ('Tuổi tại mốc đầu', pd.cut(one.age, bins=range(50, 101, 5)).astype(str)),
                          ('ADAS-Cog13 tại mốc đầu', pd.cut(one.adas13, bins=range(0, 61, 5)).astype(str))]:
        st.subheader(title)
        st.bar_chart(series.dropna().value_counts().sort_index().rename('Số lượng'))
    ev = data[np.isfinite(data.R)]
    st.subheader('Ngày ghi nhận Dementia kể từ mốc (khoảng 60 ngày)')
    st.bar_chart(pd.cut(ev.R, bins=range(0, 1141, 60)).astype(str).value_counts().sort_index().rename('Số ca'))
    st.caption('Các cụm trùng lịch tái khám: lý do mô hình dùng kết cục kiểm duyệt khoảng.')
