"""ADNI v2: mốc landmark nhiều lần, kết cục kiểm duyệt khoảng, đặc trưng tại thời điểm mốc.

Mọi đặc trưng tại mốc s chỉ dùng dữ liệu có ngày <= s. Các tham số tiền xử lý
(hiệu chỉnh ICV, tham số mô hình hỗn hợp cho slope) không dùng kết cục và được
ước lượng một lần trên train, sau đó khóa và áp dụng cho test.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
WINDOW = 1095            # cắt hành chính theo dõi tại mỗi mốc (ngày)
HORIZONS = [365, 730, 1095]
PRIMARY = 730
MRI_MAX_AGE = 365        # MRI hiện tại phải trong 365 ngày trước mốc
COG_MAX_AGE = 45         # điểm nhận thức trong 45 ngày trước mốc
MIN_SPAN = 300           # slope cần dữ liệu trải >= 300 ngày
TWO_POINT_MIN = 90       # tốc độ hai điểm kiểu v1 (phân tích độ nhạy S2)

BASE = ['tau', 'tau2']   # thời gian từ MCI đầu tiên đến mốc (năm) - số hạng landmark
MODELS = {
    'M0': BASE + ['age', 'sex', 'education', 'mmse', 'faq', 'adas13', 'ravlt_delay'],
    'M1': ['hippo_adj', 'entor', 'field3t'],
    'M2': ['adas13_slope', 'faq_slope'],
    'M3': ['hippo_slope', 'entor_slope'],
}
for prev, name in zip(['M0', 'M1', 'M2'], ['M1', 'M2', 'M3']):
    MODELS[name] = MODELS[prev] + MODELS[name]

LABELS = {
    'tau': 'Thời gian từ MCI đầu tiên (năm)', 'tau2': 'Bình phương thời gian từ MCI đầu tiên',
    'age': 'Tuổi (năm)', 'sex': 'Giới ADNI (1 nam, 2 nữ)', 'education': 'Học vấn (năm)',
    'mmse': 'MMSE (điểm)', 'faq': 'FAQ (điểm)', 'adas13': 'ADAS-Cog13 (điểm)',
    'ravlt_delay': 'RAVLT nhớ lại sau 30 phút (từ)',
    'hippo_adj': 'Hồi hải mã hiệu chỉnh ICV (mm³)', 'entor': 'Độ dày nội khứu (mm)',
    'field3t': 'MRI 3T (1) hay 1,5T (0)',
    'adas13_slope': 'Slope ADAS-Cog13 (điểm/năm)', 'faq_slope': 'Slope FAQ (điểm/năm)',
    'hippo_slope': 'Slope hồi hải mã (mm³/năm)', 'entor_slope': 'Slope nội khứu (mm/năm)',
}


# --------------------------------------------------------------------------- đọc dữ liệu
def read(root, name):
    files = list(Path(root).rglob(name + '.csv'))
    if len(files) != 1:
        raise ValueError(f'Cần đúng một {name}.csv, tìm thấy {len(files)}')
    return pd.read_csv(files[0], encoding='latin1', low_memory=False)


def has_table(root, name):
    return len(list(Path(root).rglob(name + '.csv'))) == 1


def dated(df, column):
    return df.assign(day=pd.to_datetime(df[column], errors='coerce')).dropna(subset=['day'])


# --------------------------------------------------------------------------- chẩn đoán
def visits(dx):
    """Lượt chẩn đoán hợp lệ, một dòng cho mỗi RID-ngày; loại RID-ngày có nhãn xung đột."""
    dx = dated(dx, 'EXAMDATE')
    dx['screen'] = (dx.VISCODE.fillna('').astype(str).str.lower().str.startswith('sc') |
                    dx.VISCODE2.fillna('').astype(str).str.lower().str.startswith('sc'))
    v = dx[dx.DIAGNOSIS.isin([1, 2, 3])]
    n = v.groupby(['RID', 'day']).DIAGNOSIS.nunique()
    bad = n[n > 1].reset_index()[['RID', 'day']]
    key = v.set_index(['RID', 'day']).index
    v = v[~key.isin(pd.MultiIndex.from_frame(bad))]
    out = v.groupby(['RID', 'day']).agg(dx=('DIAGNOSIS', 'first'), screen=('screen', 'all')).reset_index()
    return out.sort_values(['RID', 'day']).reset_index(drop=True), bad


def landmarks(vis, confirmed=False):
    """Mọi lượt MCI (ngoài screening) chưa từng có Dementia và còn ít nhất một lượt sau đó.

    Kết cục kiểm duyệt khoảng (L, R] tính bằng ngày kể từ mốc, cắt hành chính tại WINDOW.
    confirmed=True: chỉ tính Dementia không bị 'quay lại' MCI/CN ở các lượt sau (S4).
    """
    rows = []
    for rid, h in vis.groupby('RID', sort=False):
        days, dxs, scr = h.day.to_numpy(), h.dx.to_numpy(), h.screen.to_numpy()
        dem = np.where(dxs == 3)[0]
        if confirmed:
            dem = [i for i in dem if (dxs[i:] == 3).all()]
        first_dem = dem[0] if len(dem) else None
        first_mci = next((i for i in range(len(dxs)) if dxs[i] == 2 and not scr[i]), None)
        if first_mci is None:
            continue
        any_dem_before = np.cumsum(dxs == 3)
        for i in range(len(dxs)):
            if dxs[i] != 2 or scr[i] or any_dem_before[i] > 0 or i == len(dxs) - 1:
                continue
            s = days[i]
            after = np.arange(i + 1, len(dxs))
            if first_dem is not None and first_dem > i:
                R = (days[first_dem] - s) / np.timedelta64(1, 'D')
                prev = [j for j in after if j < first_dem]
                L = (days[prev[-1]] - s) / np.timedelta64(1, 'D') if prev else 0.0
            else:
                R = np.inf
                L = (days[after[-1]] - s) / np.timedelta64(1, 'D')
            if R > WINDOW:          # cắt hành chính
                R = np.inf
                L = min(L, WINDOW)
            if L <= 0 and not np.isfinite(R):
                continue
            tau = (s - days[first_mci]) / np.timedelta64(1, 'D') / 365.25
            mci_idx = np.where((dxs == 2) & ~scr)[0]
            second = int(len(mci_idx) > 1 and i == mci_idx[1])
            rows.append((int(rid), pd.Timestamp(s), float(L), float(R), float(tau), second))
    lm = pd.DataFrame(rows, columns=['RID', 't0', 'L', 'R', 'tau', 'second_mci'])
    lm['event'] = np.isfinite(lm.R).astype(int)
    lm['tau2'] = lm.tau ** 2
    return lm


# --------------------------------------------------------------------------- chuỗi đo lường
def _clean_score(d, col, lo, hi):
    val = pd.to_numeric(d[col], errors='coerce')
    good = val.between(lo, hi)
    if 'DONE' in d:
        good &= ~d.DONE.eq(0)
    if 'HAS_QC_ERROR' in d:
        good &= ~d.HAS_QC_ERROR.astype(str).str.lower().isin(['1', '1.0', 'true', 'yes'])
    return d[good].assign(value=val[good])[['RID', 'day', 'value']]


def clinical_series(root):
    spec = [('MMSE', 'MMSCORE', 'mmse', 0, 30), ('FAQ', 'FAQTOTAL', 'faq', 0, 30),
            ('ADAS', 'TOTAL13', 'adas13', 0, 85), ('NEUROBAT', 'AVDEL30MIN', 'ravlt_delay', 0, 15)]
    out = {}
    for table, col, name, lo, hi in spec:
        if not has_table(root, table):
            continue
        d = read(root, table)
        date_col = 'VISDATE' if 'VISDATE' in d else 'EXAMDATE'
        s = _clean_score(dated(d, date_col), col, lo, hi)
        out[name] = s.groupby(['RID', 'day'], as_index=False).value.mean()
    return out


def mri_scans(root):
    m = dated(read(root, 'UCSFFSX7'), 'EXAMDATE')
    cols = ['ST29SV', 'ST88SV', 'ST24TA', 'ST83TA', 'ST10CV']
    m[cols] = m[cols].apply(pd.to_numeric, errors='coerce')
    keep = (~m.OVERALLQC.isin(['Fail', 'Hippocampus Only']) & ~m.HIPPOQC.eq('Fail') &
            ~m.TEMPQC.eq('Fail') & m[cols[:4]].gt(0).all(axis=1))
    m = m[keep].copy()
    m['hippo'] = m.ST29SV + m.ST88SV
    m['entor'] = (m.ST24TA + m.ST83TA) / 2
    m['icv'] = m.ST10CV
    m['field3t'] = m.FIELD_STRENGTH.astype(str).str.startswith('3').astype(int)
    m['version'] = m.FSVER.astype(str) + '|' + m.FIELD_STRENGTH.astype(str)
    m['qc_done'] = m.OVERALLQC.isin(['Pass', 'Partial']).astype(int)
    m = m.sort_values(['RID', 'day']).drop_duplicates(['RID', 'day'], keep='last')
    return m[['RID', 'day', 'hippo', 'entor', 'icv', 'field3t', 'version', 'qc_done']]


def bsi_series(root):
    """BSI hồi hải mã (FOXLABBSI), năm hóa so với scan nền; dương = mất thể tích (ml)."""
    if not has_table(root, 'FOXLABBSI'):
        return pd.DataFrame(columns=['RID', 'day', 'hbsi_rate'])
    f = dated(read(root, 'FOXLABBSI'), 'EXAMDATE')
    base = f[['LONIUID', 'day']].drop_duplicates('LONIUID').rename(
        columns={'LONIUID': 'LONIUID_BASE', 'day': 'base_day'})
    h = f[f.HBSI_L.notna() & f.HBSI_R.notna() & f.HPACCEPT_L.eq(1) & f.HPACCEPT_R.eq(1)]
    h = h.merge(base, on='LONIUID_BASE', how='inner')
    dt = (h.day - h.base_day).dt.days
    h = h[dt >= TWO_POINT_MIN].assign(dt=dt[dt >= TWO_POINT_MIN])
    # mm³/năm, dấu âm = teo (cùng chiều với slope thể tích)
    h['hbsi_rate'] = -(h.HBSI_L + h.HBSI_R) * 1000 * 365.25 / h.dt
    return h.groupby(['RID', 'day'], as_index=False).hbsi_rate.mean()


# --------------------------------------------------------------------------- tiền xử lý không dùng kết cục
def icv_params(scans, vis):
    """Hệ số b của HV theo ICV, ước lượng trên người luôn CN (phương pháp phần dư)."""
    cn = vis.groupby('RID').dx.agg(lambda s: (s == 1).all())
    d = scans[scans.RID.isin(cn[cn].index) & scans.icv.notna()]
    d = d.sort_values('day').drop_duplicates('RID')     # một scan/người
    b = np.polyfit(d.icv, d.hippo, 1)[0]
    return dict(b=float(b), icv_mean=float(d.icv.mean()), n=int(len(d)))


def lme_params(series, ids, min_n=3, min_span_days=365):
    """Ước lượng moment cho mô hình hỗn hợp y = (g0+u0) + (g1+u1) t + e, t tính bằng năm."""
    coefs, inv_xtx, rss, dof = [], [], 0.0, 0
    for rid, g in series[series.RID.isin(ids)].groupby('RID'):
        if len(g) < min_n:
            continue
        t = (g.day - g.day.iloc[0]).dt.days.to_numpy() / 365.25
        if t[-1] * 365.25 < min_span_days:
            continue
        X = np.c_[np.ones_like(t), t]
        y = g.value.to_numpy(float)
        xtx = X.T @ X
        beta = np.linalg.solve(xtx, X.T @ y)
        coefs.append(beta)
        inv_xtx.append(np.linalg.inv(xtx))
        rss += float(((y - X @ beta) ** 2).sum())
        dof += len(y) - 2
    coefs = np.array(coefs)
    sigma2 = rss / dof
    D = np.cov(coefs.T) - sigma2 * np.mean(inv_xtx, axis=0)
    w, V = np.linalg.eigh(D)
    D = V @ np.diag(np.maximum(w, 1e-6 * max(w.max(), 1e-12))) @ V.T
    return dict(gamma=coefs.mean(0).tolist(), D=D.tolist(), sigma2=sigma2, n=int(len(coefs)))


def blup_slope(t_years, y, p):
    X = np.c_[np.ones_like(t_years), t_years]
    g, D = np.array(p['gamma']), np.array(p['D'])
    V = X @ D @ X.T + p['sigma2'] * np.eye(len(y))
    u = D @ X.T @ np.linalg.solve(V, y - X @ g)
    return float(g[1] + u[1])


def fit_preprocessing(root):
    vis, _ = visits(read(root, 'DXSUM'))
    scans = mri_scans(root)
    clin = clinical_series(root)
    icv = icv_params(scans, vis)
    mci_ids = set(vis.loc[vis.dx == 2, 'RID'])
    scans = scans.assign(hippo_adj=scans.hippo - icv['b'] * (scans.icv - icv['icv_mean']))
    params = dict(icv=icv)
    for name, col in [('hippo_adj', 'hippo_adj'), ('entor', 'entor')]:
        s = scans[['RID', 'day', col, 'version']].rename(columns={col: 'value'})
        # chỉ ước lượng trên chuỗi cùng phiên bản/cường độ từ trường
        s = s[s.groupby('RID').version.transform(lambda v: v == v.mode().iloc[0])]
        params[name] = lme_params(s, mci_ids)
    for name in ['adas13', 'faq']:
        params[name] = lme_params(clin[name], mci_ids)
    return params


# --------------------------------------------------------------------------- đặc trưng tại mốc
def demographics(root):
    d = read(root, 'PTDEMOG')
    out = {}
    for rid, g in d.groupby('RID'):
        r = {}
        for col, key, lo, hi in [('PTEDUCAT', 'education', 0, 36), ('PTGENDER', 'sex', 1, 2)]:
            s = pd.to_numeric(g[col], errors='coerce')
            s = s[s.between(lo, hi)]
            if len(s):
                r[key] = float(s.iloc[-1])
        years = pd.to_numeric(g.PTDOBYY.astype(str).str.extract(r'((?:18|19|20)\d{2})')[0], errors='coerce')
        if 'PTDOB' in g:
            years = years.fillna(pd.to_numeric(g.PTDOB.astype(str).str.extract(r'((?:18|19|20)\d{2})')[0], errors='coerce'))
        years = years.dropna().unique()
        if len(years) == 1:
            r['birth_year'] = float(years[0])
        out[int(rid)] = r
    return out


def build(root, params, confirmed=False):
    """Bảng mốc landmark với đặc trưng as-of và kết cục (L, R]."""
    vis, conflicts = visits(read(root, 'DXSUM'))
    lm = landmarks(vis, confirmed=confirmed)
    demo = demographics(root)
    clin = {k: dict(tuple(v.groupby('RID'))) for k, v in clinical_series(root).items()}
    scans = mri_scans(root)
    icv = params['icv']
    scans['hippo_adj'] = scans.hippo - icv['b'] * (scans.icv - icv['icv_mean'])
    scans_by = dict(tuple(scans.groupby('RID')))
    bsi = dict(tuple(bsi_series(root).groupby('RID')))
    feats = []
    for r in lm.itertuples():
        f = {}
        d = demo.get(r.RID, {})
        f['sex'], f['education'] = d.get('sex', np.nan), d.get('education', np.nan)
        by = d.get('birth_year')
        f['age'] = r.t0.year - by if by and 18 <= r.t0.year - by <= 115 else np.nan
        for name, groups in clin.items():
            g = groups.get(r.RID)
            if g is None:
                continue
            g = g[g.day <= r.t0]
            if len(g) and (r.t0 - g.day.iloc[-1]).days <= COG_MAX_AGE:
                f[name] = g.value.iloc[-1]
            if name in ('adas13', 'faq') and len(g) >= 2:
                t = (g.day - g.day.iloc[0]).dt.days.to_numpy() / 365.25
                if t[-1] * 365.25 >= MIN_SPAN and (r.t0 - g.day.iloc[-1]).days <= COG_MAX_AGE:
                    f[name + '_slope'] = blup_slope(t, g.value.to_numpy(float), params[name])
        g = scans_by.get(r.RID)
        if g is not None:
            g = g[g.day <= r.t0]
            if len(g) and (r.t0 - g.day.iloc[-1]).days <= MRI_MAX_AGE:
                cur = g.iloc[-1]
                f.update(hippo=cur.hippo, hippo_adj=cur.hippo_adj, entor=cur.entor, field3t=cur.field3t,
                         mri_age_days=(r.t0 - cur.day).days, qc_done=cur.qc_done)
                same = g[g.version == cur.version]
                span = (same.day.iloc[-1] - same.day.iloc[0]).days
                f['mri_span_days'] = span
                if span >= MIN_SPAN:
                    t = (same.day - same.day.iloc[0]).dt.days.to_numpy() / 365.25
                    f['hippo_slope'] = blup_slope(t, same.hippo_adj.to_numpy(float), params['hippo_adj'])
                    f['entor_slope'] = blup_slope(t, same.entor.to_numpy(float), params['entor'])
                old = same[same.day <= cur.day - pd.Timedelta(days=TWO_POINT_MIN)]
                if len(old):   # tốc độ hai điểm kiểu v1 (S2)
                    o = old.iloc[-1]
                    dd = (cur.day - o.day).days
                    f['hippo_rate2'] = 365.25 * (cur.hippo - o.hippo) / dd
                    f['entor_rate2'] = 365.25 * (cur.entor - o.entor) / dd
                    f['pair_days'] = dd
        b = bsi.get(r.RID)
        if b is not None:
            b = b[b.day <= r.t0]
            if len(b) and (r.t0 - b.day.iloc[-1]).days <= MRI_MAX_AGE:
                f['hbsi_rate'] = b.hbsi_rate.iloc[-1]
        feats.append(f)
    out = pd.concat([lm.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    for col in set(sum(MODELS.values(), [])) | {'hippo_rate2', 'entor_rate2', 'hbsi_rate'}:
        if col not in out:
            out[col] = np.nan
    return out, conflicts


def save_json(obj, path):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, allow_nan=False, ensure_ascii=False), encoding='utf8')
    tmp.replace(path)
