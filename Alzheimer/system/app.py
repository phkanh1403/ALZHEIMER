"""Ứng dụng Streamlit ver5: dự báo nguy cơ sa sút trí tuệ tại một lần tái khám MCI."""
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
from registry import initialize, import_zip, BOUNDS, FIELDS, upsert

st.set_page_config(page_title='ADNI MCI ver5', layout='wide')
R = ROOT / 'runs'
prep = json.loads((R / 'preprocessing.json').read_text())
SETS = {
    'secondary': dict(label='Dự báo theo dữ liệu hiện có (khuyến nghị)', models=json.loads((R / 'secondary_models.json').read_text()),
                      boot='bootstrap_secondary.json', note='Mỗi mô hình học trên mọi mốc đủ biến của riêng nó.'),
    'paired': dict(label='So sánh M0–M3 (bộ ghép cặp)', models=json.loads((R / 'models.json').read_text()),
                   boot='bootstrap_paired.json', note='Cả bốn mô hình học trên cùng tập mốc đủ biến M3; dùng để đánh giá giá trị gia tăng.'),
}
inputs = pd.read_csv(R / 'train_inputs.csv', parse_dates=['t0'])
MRI_RULE = f'≥ 2 MRI T1 cùng phiên bản FreeSurfer và cùng cường độ từ trường, trải ≥ {MIN_SPAN} ngày'


@st.cache_data
def load_json(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


def table(path):
    p = R / path
    return pd.read_csv(p) if p.exists() else None


def perf(set_key, model, h):
    """Hiệu năng của CHÍNH bộ mô hình đang dùng: CV (train) và test."""
    if set_key == 'paired':
        cv, te = table('cv/main_metrics.csv'), table('test_metrics.csv')
        cv = cv[(cv.scope == 'random') & (cv.model == model) & (cv.h == h)] if cv is not None else None
        te = te[(te.scope == 'random') & (te.model == model) & (te.h == h)] if te is not None else None
    else:
        cv, te = table(f'cv/SEC_{model}_metrics.csv'), table('test_coverage.csv')
        cv = cv[(cv.scope == 'random') & (cv.h == h)] if cv is not None else None
        te = te[(te.scope == 'random') & (te.model == model) & (te.h == h)] if te is not None else None
    f = lambda d: f'{d.auc.iloc[0]:.3f}' if d is not None and len(d) else '—'
    return f(cv), f(te)


db = sqlite3.connect(ROOT / 'profiles.sqlite', check_same_thread=False)
db.execute('CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0)')
initialize(db, inputs)
profiles = pd.read_sql('SELECT * FROM profiles ORDER BY star DESC,name,day', db)

st.title('Theo dõi MCI · ver5')
st.caption(f'Phạm vi: người MCI CHƯA dùng kháng thể kháng amyloid. Mô hình M3 chỉ áp dụng khi có {MRI_RULE}. '
           'Công cụ hỗ trợ phân tầng theo dõi, không dùng để chẩn đoán hay quyết định điều trị.')
page = st.sidebar.radio('Chức năng', ['Dự đoán', 'Nhập hồ sơ', 'Hồ sơ đã lưu', 'Dashboard mô hình', 'Kiểm toán dữ liệu'])


def risk_table(row, h, set_key):
    S = SETS[set_key]
    draws = load_json(S['boot'])
    span = row.get('mri_span_days')
    out = []
    for name in MODELS:
        m = S['models'][name]
        missing = [LABELS[c] for c in MODELS[name] if row.get(c) is None or pd.isna(row.get(c))]
        cv_auc, test_auc = perf(set_key, name, h)
        base = {'Mô hình': name, 'Bộ': m.get('set', ''), 'Số người train': m.get('persons_train', ''),
                f'AUC CV ({h} ngày)': cv_auc, f'AUC test ({h} ngày)': test_auc}
        if name == 'M3' and (span is None or pd.isna(span) or span < MIN_SPAN):
            out.append({**base, 'Nguy cơ': f'Không áp dụng: khoảng đo MRI {"chưa có" if span is None or pd.isna(span) else f"{int(span)} ngày"} (< {MIN_SPAN})',
                        'Khoảng bootstrap 95%': '—'})
            continue
        if missing:
            out.append({**base, 'Nguy cơ': 'Thiếu: ' + ', '.join(missing), 'Khoảng bootstrap 95%': '—'})
            continue
        frame = pd.DataFrame([{c: row[c] for c in MODELS[name]}])
        p = predict(m, frame, h)[0]
        ci = '—'
        if name in draws:
            q = np.quantile([predict(d, frame, h)[0] for d in draws[name]], [.025, .975])
            ci = f'{q[0]:.1%} – {q[1]:.1%}'
        out.append({**base, 'Nguy cơ': f'{p:.1%}', 'Khoảng bootstrap 95%': ci})
    return pd.DataFrame(out)


def provenance(row):
    keys = [('mri_date', 'Ngày chụp MRI hiện tại'), ('mri_from', 'Chuỗi MRI tính slope bắt đầu từ'),
            ('mri_span_days', 'Khoảng đo MRI (ngày)'), ('scanner', 'Máy chụp (hãng/model)'),
            ('mmse_date', 'Ngày MMSE'), ('adas13_date', 'Ngày ADAS-Cog13'), ('adas13_from', 'Chuỗi ADAS từ'), ('faq_from', 'Chuỗi FAQ từ')]
    return {lab: row.get(k) for k, lab in keys if row.get(k) is not None and not (isinstance(row.get(k), float) and np.isnan(row.get(k)))}


if page == 'Dự đoán':
    mode = st.radio('Chế độ', list(SETS), format_func=lambda k: SETS[k]['label'], horizontal=True)
    st.caption(SETS[mode]['note'] + (' Chênh lệch giữa các mô hình trên MỘT người không phải bằng chứng về giá trị gia tăng.' if mode == 'paired' else ''))
    src = st.radio('Nguồn', ['ADNI train', 'Hồ sơ đã lưu'], horizontal=True)
    if src == 'ADNI train':
        rid = st.selectbox('RID', sorted(inputs.RID.unique()))
        pts = inputs[inputs.RID == rid].sort_values('t0')
        dates = pts.t0.dt.date.astype(str).tolist()
        when = st.selectbox('Mốc (lượt MCI; mặc định lượt mới nhất)', dates, index=len(dates) - 1)
        row = pts[pts.t0.dt.date.astype(str) == when].iloc[0].to_dict()
        day = when
    else:
        usable = profiles[profiles.day != '']
        if usable.empty:
            st.info('Chưa có hồ sơ.')
            st.stop()
        ix = st.selectbox('Hồ sơ', usable.index, format_func=lambda i: usable.loc[i, 'name'] + ' · ' + usable.loc[i, 'day'])
        day, row = usable.loc[ix, 'day'], json.loads(usable.loc[ix, 'features'])
    if row.get('has_followup') == 0:
        st.info('Đây là lượt MCI mới nhất (chưa có lần khám sau) — vẫn dự báo được.')
    h = st.selectbox('Chân trời (ngày)', HORIZONS, index=HORIZONS.index(PRIMARY))
    st.write('Ngày mốc', day, '· Ngày đích', (pd.Timestamp(day) + pd.Timedelta(days=h)).date())
    with st.expander('Nguồn gốc số đo'):
        st.write(provenance(row))
    if st.checkbox('Bệnh nhân đang dùng thuốc kháng amyloid (lecanemab, donanemab…)'):
        st.error('Ngoài phạm vi: điều trị kháng amyloid gây giả teo (ARPA) làm sai lệch tốc độ teo. Không hiển thị dự báo.')
    elif st.button('Dự đoán'):
        st.dataframe(risk_table(row, h, mode), hide_index=True)
        st.caption('Xác suất ghi nhận Dementia đến ngày đích, giả định người bệnh còn sống. Khoảng bootstrap (200 lần, theo cụm người) '
                   'phản ánh biến thiên do lấy mẫu train của CHÍNH bộ mô hình đang dùng. AUC CV/test là hiệu năng của cùng bộ đó.')

elif page == 'Nhập hồ sơ':
    manual, batch = st.tabs(['Nhập từng hồ sơ', 'Nhập ZIP ADNI'])
    with batch:
        up = st.file_uploader('ZIP chứa DXSUM, MMSE, FAQ, ADAS, NEUROBAT, PTDEMOG, UCSFFSX7 '
                              '(tùy chọn: FOXLABBSI, All_Subjects_Structural_MRI_Images)', type='zip')
        if up and st.button('Lưu hồ sơ từ ZIP'):
            try:
                r = import_zip(db, up, prep, set(inputs.RID))
                st.success(f"Thêm {r['added']} mốc mới, cập nhật {r['updated']} mốc (khóa RID + ngày mốc). "
                           f"{r['rids']} RID trong ZIP, {r['skipped_train']} RID trùng train bị bỏ qua. Mỗi RID lấy lượt MCI mới nhất.")
            except Exception as exc:
                st.error(f'Không thể nhập ZIP: {exc}')
    with manual:
        st.caption('Nhập tại một lần tái khám MCI. Có thể để trống phép đo chưa có.')
        with st.expander('Tính slope từ chuỗi đo (tham số khóa từ train)'):
            st.caption(f'Mỗi dòng: YYYY-MM-DD,giá trị. Hồi hải mã: thể tích đã hiệu chỉnh ICV '
                       f'(HV_adj = HV − {prep["icv"]["b"]:.5f} × (ICV − {prep["icv"]["icv_mean"]:.0f})). '
                       'Các lần chụp MRI phải cùng phiên bản FreeSurfer và cùng cường độ từ trường.')
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
                        st.success(f'Slope (BLUP): {blup_slope(t, pts.v.astype(float).to_numpy(), prep[kind]):.3f} đơn vị/năm · '
                                   f'khoảng đo {int(t[-1] * 365.25)} ngày')
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
            span = st.number_input(f'Khoảng đo MRI dùng tính slope (ngày, bắt buộc cho M3; ≥ {MIN_SPAN})', min_value=0, max_value=10000, value=None)
            if st.form_submit_button('Lưu hồ sơ'):
                if not name.strip() or not eligible:
                    st.error('Cần mã hồ sơ và xác nhận điều kiện tại mốc.')
                else:
                    if (feats.get('hippo_slope') is not None or feats.get('entor_slope') is not None) and (span is None or span < MIN_SPAN):
                        st.warning(f'Đã lưu nhưng M3 sẽ bị chặn khi dự báo: khoảng đo MRI chưa có hoặc < {MIN_SPAN} ngày.')
                    feats['tau2'] = None if feats['tau'] is None else feats['tau'] ** 2
                    feats['mri_span_days'] = span
                    with db:
                        upsert(db, [(f'Manual:{uuid.uuid4()}', name.strip(), str(day), json.dumps(feats))])
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
    summ = load_json('summary.json')
    if summ:
        st.subheader('Kết luận theo tầng (quy tắc khóa trước, δ = %.2f, biên NB = %.2f)' % (summ['protocol']['delta_auc'], summ['protocol']['nb_margin']))
        rows = []
        for h, t in summ['tiers'].items():
            for k, lab in [('cv', 'CV lặp trên train 80%'), ('test', 'Test 20% (kiểm chứng nội bộ)')]:
                x = t[k]
                rows.append({'Chân trời': h, 'Tầng': lab, 'ΔAUC (M3 − M2)': f"{x['d_auc']:.4f} ({x['auc_ci'][0]:.4f}; {x['auc_ci'][1]:.4f})",
                             'ΔNB 15–30%': f"{x['d_nb']:.4f} ({x['nb_ci'][0]:.4f}; {x['nb_ci'][1]:.4f})", 'Theo quy tắc': x['conclusion']})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.info('Tổng kết: ' + summ['overall'] + '. ' + summ['test_note'])
    cov = table('coverage.csv')
    if cov is not None:
        st.subheader('Độ bao phủ train')
        st.dataframe(cov, hide_index=True)
    mm = table('cv/main_metrics.csv')
    if mm is not None:
        st.subheader('CV lặp 5 fold × 20 (bộ ghép cặp, một mốc ngẫu nhiên/người)')
        v = mm[mm.scope == 'random']
        st.line_chart(v.pivot(index='h', columns='model', values='auc').rename_axis('Chân trời (ngày)'))
        st.dataframe(v[['h', 'model', 'case_certain', 'control_certain', 'ambiguous', 'censored_early', 'auc', 'ipa',
                        'oe_turnbull', 'cal_slope_turnbull', 'nb_core']].round(4), hide_index=True)
        st.caption('Bốn nhóm trạng thái tại chân trời: ca chắc chắn (R ≤ h), chắc chắn chưa biến cố (L ≥ h), khoảng vắt qua ngày đích '
                   '(L < h < R; rút T từ M0), ngừng theo dõi sớm (trọng số 0, hiệu chỉnh IPCW). O/E và slope hiệu chuẩn tính bằng Turnbull (10 nhóm).')
    d = table('cv/main_deltas.csv')
    if d is not None:
        st.subheader('Chênh lệch ghép cặp (CV)')
        st.dataframe(d.round(4), hide_index=True)
    s = table('sensitivity_summary.csv')
    if s is not None:
        st.subheader('Phân tích độ nhạy (thăm dò, 730 ngày)')
        st.dataframe(s.round(4), hide_index=True)
    tm = table('test_metrics.csv')
    if tm is not None:
        st.subheader('Test 20%')
        st.dataframe(tm[tm.scope == 'random'][['h', 'model', 'case_certain', 'control_certain', 'ambiguous', 'censored_early',
                                               'auc', 'ipa', 'oe_turnbull', 'nb_core']].round(4), hide_index=True)
        st.dataframe(table('test_deltas.csv').round(4), hide_index=True)
        cal = table('test_calibration.csv')
        hh = st.selectbox('Hiệu chuẩn test tại chân trời', HORIZONS, index=1)
        st.scatter_chart(cal[(cal.h == hh) & cal.model.isin(['M2', 'M3'])], x='predicted', y='observed_turnbull', color='model')
    up = st.file_uploader('Chấm lại bằng ZIP test', type='zip')
    if up and st.button('Đánh giá bộ test'):
        from evaluate import evaluate
        try:
            with tempfile.TemporaryDirectory() as folder, zipfile.ZipFile(up) as archive:
                for i in archive.infolist():
                    if i.filename.endswith('.csv'):
                        archive.extract(i, folder)
                with st.spinner('Đang dựng mốc và đánh giá (vài phút)'):
                    m, dl, info = evaluate(folder)
                st.write(info)
                st.dataframe(dl.round(4), hide_index=True)
        except Exception as err:
            st.error(str(err))

else:
    st.header('Kiểm toán dữ liệu train')
    audit = load_json('data_audit.json')
    st.json(audit)
    c = table('conflicts.csv')
    if c is not None:
        st.subheader(f'Xung đột đã ghi nhận ({len(c)})')
        st.dataframe(c.groupby(['table', 'action']).size().rename('Số bản ghi').reset_index(), hide_index=True)
        st.download_button('Tải conflicts.csv', c.to_csv(index=False).encode('utf-8'), 'conflicts.csv')
    lm = pd.read_csv(R / 'train_landmarks.csv')
    ev = lm[np.isfinite(lm.R)]
    st.subheader('Ngày ghi nhận Dementia kể từ mốc (60 ngày/khoảng)')
    st.bar_chart(pd.cut(ev.R, bins=range(0, 1861, 60)).astype(str).value_counts().sort_index().rename('Số ca'))
