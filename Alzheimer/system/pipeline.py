"""ADNI ver5: nguồn dữ liệu sạch, điểm mốc, nhãn kiểm duyệt khoảng và đặc trưng tại thời điểm mốc.

Tách ba bước:
  1. Sources        — đọc bảng, xử lý trùng/xung đột (ghi conflicts), thống kê sẵn cho tham số.
  2. raw_points()   — đầu vào DỰ BÁO tại mỗi lượt MCI (kể cả lượt mới nhất), không cần lượt sau.
     attach_labels() — nhãn (L, R] chỉ cho các mốc có lượt chẩn đoán sau đó (train/test).
  3. fit_params() / apply_params() — hiệu chỉnh ICV và tham số BLUP, ước lượng trên một tập RID
     cho trước (trong từng fold) rồi áp cho mọi mốc.
Mọi đặc trưng tại mốc s chỉ dùng dữ liệu có ngày <= s.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

WINDOW = 1825            # cửa sổ khớp mô hình: theo dõi bị kiểm duyệt tại 1.825 ngày
EVAL_MAX = 1095          # chân trời đánh giá tối đa
HORIZONS = [365, 730, 1095]
PRIMARY = 730
MRI_MAX_AGE = 365
COG_MAX_AGE = 45
MIN_SPAN = 300
TWO_POINT_MIN = 90

BASE = ['tau', 'tau2']
MODELS = {
    'M0': BASE + ['age', 'sex', 'education', 'mmse', 'faq', 'adas13', 'ravlt_delay'],
    'M1': ['hippo_adj', 'entor', 'field3t'],
    'M2': ['adas13_slope', 'faq_slope'],
    'M3': ['hippo_slope', 'entor_slope'],
}
for _p, _n in zip(['M0', 'M1', 'M2'], ['M1', 'M2', 'M3']):
    MODELS[_n] = MODELS[_p] + MODELS[_n]

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
SERIES = ['adas13', 'faq', 'mri']


def read(root, name):
    files = list(Path(root).rglob(name + '.csv'))
    if len(files) != 1:
        raise ValueError(f'Cần đúng một {name}.csv, tìm thấy {len(files)}')
    return pd.read_csv(files[0], encoding='latin1', low_memory=False)


def has_table(root, name):
    return len(list(Path(root).rglob(name + '.csv'))) == 1


def dated(df, column):
    return df.assign(day=pd.to_datetime(df[column], errors='coerce')).dropna(subset=['day'])


def years(a, b):
    return (a - b) / np.timedelta64(1, 'D') / 365.25


# =========================================================================== chẩn đoán
def visits(dx):
    dx = dated(dx, 'EXAMDATE')
    dx['screen'] = (dx.VISCODE.fillna('').astype(str).str.lower().str.startswith('sc') |
                    dx.VISCODE2.fillna('').astype(str).str.lower().str.startswith('sc'))
    v = dx[dx.DIAGNOSIS.isin([1, 2, 3])]
    n = v.groupby(['RID', 'day']).DIAGNOSIS.nunique()
    bad = n[n > 1].reset_index()[['RID', 'day']]
    v = v[~v.set_index(['RID', 'day']).index.isin(pd.MultiIndex.from_frame(bad))]
    out = v.groupby(['RID', 'day']).agg(dx=('DIAGNOSIS', 'first'), screen=('screen', 'all')).reset_index()
    return out.sort_values(['RID', 'day']).reset_index(drop=True), bad


def eligible_points(vis):
    """Mọi lượt MCI ngoài screening chưa từng có Dementia — KỂ CẢ lượt mới nhất (đầu vào dự báo)."""
    rows = []
    for rid, h in vis.groupby('RID', sort=False):
        days, dxs, scr = h.day.to_numpy(), h.dx.to_numpy(), h.screen.to_numpy()
        mci_idx = np.where((dxs == 2) & ~scr)[0]
        if not len(mci_idx):
            continue
        seen_dem = np.cumsum(dxs == 3)
        for i in mci_idx:
            if seen_dem[i] > 0:
                continue
            rows.append((int(rid), pd.Timestamp(days[i]), years(days[i], days[mci_idx[0]]),
                         int(len(mci_idx) > 1 and i == mci_idx[1]), int(i < len(dxs) - 1)))
    p = pd.DataFrame(rows, columns=['RID', 't0', 'tau', 'second_mci', 'has_followup'])
    p['tau2'] = p.tau ** 2
    return p


def attach_labels(points, vis, confirmed=False):
    """Nhãn (L, R] cho mốc có lượt sau. Giữ cận gốc khi L < WINDOW (1.825 ngày); L >= WINDOW -> kiểm duyệt tại WINDOW."""
    by = dict(tuple(vis.groupby('RID')))
    L_, R_ = [], []
    for r in points.itertuples():
        h = by[r.RID]
        days, dxs = h.day.to_numpy(), h.dx.to_numpy()
        after = np.where(days > np.datetime64(r.t0))[0]
        if not len(after):
            L_.append(np.nan); R_.append(np.nan); continue
        dem = [j for j in after if dxs[j] == 3]
        if confirmed:
            dem = [j for j in dem if (dxs[j:] == 3).all()]
        s = np.datetime64(r.t0)
        if dem:
            R = (days[dem[0]] - s) / np.timedelta64(1, 'D')
            prev = [j for j in after if j < dem[0]]
            L = (days[prev[-1]] - s) / np.timedelta64(1, 'D') if prev else 0.0
        else:
            R, L = np.inf, (days[after[-1]] - s) / np.timedelta64(1, 'D')
        if L >= WINDOW:
            L, R = float(WINDOW), np.inf
        L_.append(float(L)); R_.append(float(R))
    out = points.copy()
    out['L'], out['R'] = L_, R_
    out = out[out.L.notna() & ~((out.L <= 0) & ~np.isfinite(out.R))].copy()
    out['event'] = np.isfinite(out.R).astype(int)
    return out.reset_index(drop=True)


# =========================================================================== nguồn dữ liệu
class Sources:
    """Đọc và làm sạch một thư mục dữ liệu; mọi xung đột được ghi vào self.conflicts."""

    def __init__(self, root):
        self.root = Path(root)
        self.conflicts = []
        self.vis, bad = visits(read(root, 'DXSUM'))
        for r in bad.itertuples():
            self.conflicts.append(dict(table='DXSUM', RID=r.RID, day=r.day.date(), detail='nhãn chẩn đoán khác nhau', action='loại lượt'))
        self.clin = self._clinical()
        self.scans, self.meas_error = self._mri()
        self.bsi = self._bsi()
        self.demo, self.demo_conflict = self._demographics()
        self._by = {k: dict(tuple(v.groupby('RID'))) for k, v in self.clin.items()}
        self._scans_by = dict(tuple(self.scans.groupby('RID')))
        self._bsi_by = dict(tuple(self.bsi.groupby('RID')))
        self._demo_by = dict(tuple(self.demo.groupby('RID')))
        self._stats = self._lme_stats()

    # ---------------------------------------------------------------- lâm sàng
    def _clinical(self):
        spec = [('MMSE', 'MMSCORE', 'mmse', 0, 30), ('FAQ', 'FAQTOTAL', 'faq', 0, 30),
                ('ADAS', 'TOTAL13', 'adas13', 0, 85), ('NEUROBAT', 'AVDEL30MIN', 'ravlt_delay', 0, 15)]
        out = {}
        for table, col, name, lo, hi in spec:
            if not has_table(self.root, table):
                continue
            d = read(self.root, table)
            d = dated(d, 'VISDATE' if 'VISDATE' in d else 'EXAMDATE')
            val = pd.to_numeric(d[col], errors='coerce')
            good = val.between(lo, hi)
            if 'DONE' in d:
                good &= ~d.DONE.eq(0)
            if 'HAS_QC_ERROR' in d:
                good &= ~d.HAS_QC_ERROR.astype(str).str.lower().isin(['1', '1.0', 'true', 'yes'])
            s = d[good].assign(value=val[good])[['RID', 'day', 'value']]
            n = s.groupby(['RID', 'day']).value.nunique()
            bad = n[n > 1].index
            for rid, day in bad:
                vals = sorted(s[(s.RID == rid) & (s.day == day)].value.unique().tolist())
                self.conflicts.append(dict(table=table, RID=rid, day=day.date(), detail=f'{col}={vals}', action='loại số đo'))
            s = s[~s.set_index(['RID', 'day']).index.isin(bad)].drop_duplicates(['RID', 'day'])
            out[name] = s.sort_values(['RID', 'day']).reset_index(drop=True)
        return out

    # ---------------------------------------------------------------- MRI
    def _mri(self):
        m = dated(read(self.root, 'UCSFFSX7'), 'EXAMDATE')
        cols = ['ST29SV', 'ST88SV', 'ST24TA', 'ST83TA', 'ST10CV']
        m[cols] = m[cols].apply(pd.to_numeric, errors='coerce')
        keep = (~m.OVERALLQC.isin(['Fail', 'Hippocampus Only']) & ~m.HIPPOQC.eq('Fail') &
                ~m.TEMPQC.eq('Fail') & m[cols[:4]].gt(0).all(axis=1))
        m = m[keep].copy()
        m['hippo'] = m.ST29SV + m.ST88SV
        m['entor'] = (m.ST24TA + m.ST83TA) / 2
        m['icv'] = m.ST10CV
        m['field'] = m.FIELD_STRENGTH.astype(str)
        m['field3t'] = m.field.str.startswith('3').astype(int)
        m['model'] = 'NA'
        if has_table(self.root, 'All_Subjects_Structural_MRI_Images'):
            a = read(self.root, 'All_Subjects_Structural_MRI_Images')[['image_id', 'mri_mfr', 'mri_mfr_model']]
            a = a.drop_duplicates('image_id').set_index('image_id')
            mm = (a.mri_mfr.astype(str) + ' ' + a.mri_mfr_model.astype(str)).to_dict()
            m['model'] = m.IMAGEUID.map(mm).fillna('NA')
        m['qc_done'] = m.OVERALLQC.isin(['Pass', 'Partial']).astype(int)
        m['run'] = pd.to_datetime(m.RUNDATE, errors='coerce')
        # cường độ từ trường phổ biến nhất của mỗi RID (hòa -> 3T)
        modal = m.groupby('RID').field3t.agg(lambda s: int(s.mean() >= 0.5))
        rows, diffs = [], []
        for (rid, day), g in m.groupby(['RID', 'day']):
            if len(g) == 1:
                rows.append(g.iloc[0]); continue
            if g.field3t.nunique() > 1:
                pick = g[g.field3t == modal[rid]]
                self.conflicts.append(dict(table='UCSFFSX7', RID=rid, day=day.date(),
                                           detail=f'cùng ngày khác từ trường {sorted(g.field.unique())}, IMAGEUID={g.IMAGEUID.tolist()}',
                                           action=f'giữ {"3T" if modal[rid] else "1.5T"} (phổ biến nhất của RID)'))
                g = pick
            if len(g) > 1:
                if g.hippo.nunique() > 1 and g.FSVER.nunique() == 1:
                    diffs.append(float(g.hippo.iloc[0] - g.hippo.iloc[1]))
                    self.conflicts.append(dict(table='UCSFFSX7', RID=rid, day=day.date(),
                                               detail=f'{len(g)} lần chụp liên tiếp cùng giao thức, HV={g.hippo.round(0).tolist()}, IMAGEUID={g.IMAGEUID.tolist()}',
                                               action='lấy trung bình'))
                base = g.iloc[0].copy()
                for c in ['hippo', 'entor', 'icv']:
                    base[c] = g[c].mean()
                base['qc_done'] = int(g.qc_done.max())
                base['model'] = g.model.mode().iloc[0]
                rows.append(base)
            else:
                rows.append(g.iloc[0])
        s = pd.DataFrame(rows)[['RID', 'day', 'hippo', 'entor', 'icv', 'field', 'field3t', 'FSVER', 'model', 'qc_done', 'IMAGEUID', 'run']]
        s['version'] = s.FSVER.astype(str) + '|' + s.field
        s['version_model'] = s.version + '|' + s.model
        d = np.array(diffs)
        err = dict(pairs=int(len(d)),
                   median_abs_diff=float(np.median(np.abs(d))) if len(d) else np.nan,
                   sigma_robust=float(np.median(np.abs(d)) / 0.6745 / np.sqrt(2)) if len(d) else np.nan)
        return s.sort_values(['RID', 'day']).reset_index(drop=True), err

    def _bsi(self):
        if not has_table(self.root, 'FOXLABBSI'):
            return pd.DataFrame(columns=['RID', 'day', 'hbsi_rate'])
        f = dated(read(self.root, 'FOXLABBSI'), 'EXAMDATE')
        base = f[['LONIUID', 'day']].drop_duplicates('LONIUID').rename(columns={'LONIUID': 'LONIUID_BASE', 'day': 'base_day'})
        h = f[f.HBSI_L.notna() & f.HBSI_R.notna() & f.HPACCEPT_L.eq(1) & f.HPACCEPT_R.eq(1)].merge(base, on='LONIUID_BASE')
        dt = (h.day - h.base_day).dt.days
        h = h[dt >= TWO_POINT_MIN].assign(dt=dt[dt >= TWO_POINT_MIN])
        h['hbsi_rate'] = -(h.HBSI_L + h.HBSI_R) * 1000 * 365.25 / h.dt
        return h.groupby(['RID', 'day'], as_index=False).hbsi_rate.mean()

    def _demographics(self):
        d = read(self.root, 'PTDEMOG')
        d = d.assign(day=pd.to_datetime(d.VISDATE, errors='coerce'))
        d['sex'] = pd.to_numeric(d.PTGENDER, errors='coerce').where(lambda s: s.between(1, 2))
        d['education'] = pd.to_numeric(d.PTEDUCAT, errors='coerce').where(lambda s: s.between(0, 36))
        y = pd.to_numeric(d.PTDOBYY.astype(str).str.extract(r'((?:18|19|20)\d{2})')[0], errors='coerce')
        if 'PTDOB' in d:
            y = y.fillna(pd.to_numeric(d.PTDOB.astype(str).str.extract(r'((?:18|19|20)\d{2})')[0], errors='coerce'))
        d['birth_year'] = y
        conflict = set()
        for col, label in [('sex', 'giới'), ('education', 'học vấn')]:
            n = d.dropna(subset=[col]).groupby('RID')[col].nunique()
            bad = set(n[n > 1].index.astype(int))
            conflict |= bad
            for rid in sorted(bad):
                vals = sorted(d.loc[d.RID == rid, col].dropna().unique().tolist())
                self.conflicts.append(dict(table='PTDEMOG', RID=rid, day='', detail=f'{label} khác nhau giữa các bản ghi: {vals}',
                                           action='lấy bản ghi gần nhất <= mốc; loại ở S7'))
        return d[['RID', 'day', 'sex', 'education', 'birth_year']], conflict

    # ---------------------------------------------------------------- thống kê sẵn cho tham số
    def _lme_stats(self, min_n=3, min_span_days=365):
        """OLS theo từng người cho mỗi chuỗi; tham số quần thể của một tập RID chỉ là phép gộp."""
        mci = set(self.vis.loc[self.vis.dx == 2, 'RID'])
        stats = {}

        def ols(t, ys):
            X = np.c_[np.ones_like(t), t]
            xtx_inv = np.linalg.inv(X.T @ X)
            B = xtx_inv @ X.T @ np.array(ys).T          # 2 x k
            res = np.array(ys).T - X @ B
            return B, xtx_inv, res

        for name in ['adas13', 'faq']:
            rec = {}
            for rid, g in self.clin[name][self.clin[name].RID.isin(mci)].groupby('RID'):
                if len(g) < min_n:
                    continue
                t = years(g.day.to_numpy(), g.day.to_numpy()[0])
                if t[-1] * 365.25 < min_span_days:
                    continue
                B, inv, res = ols(t, [g.value.to_numpy(float)])
                rec[rid] = dict(coef=B[:, 0], inv=inv, rss=float(res[:, 0] @ res[:, 0]), dof=len(t) - 2)
            stats[name] = rec
        rec = {}
        for rid, g in self.scans[self.scans.RID.isin(mci)].groupby('RID'):
            g = g[g.version == g.version.mode().iloc[0]]
            if len(g) < min_n or g.icv.isna().any():
                continue
            t = years(g.day.to_numpy(), g.day.to_numpy()[0])
            if t[-1] * 365.25 < min_span_days:
                continue
            B, inv, res = ols(t, [g.hippo.to_numpy(float), g.icv.to_numpy(float), g.entor.to_numpy(float)])
            rec[rid] = dict(coef_h=B[:, 0], coef_i=B[:, 1], coef_e=B[:, 2], inv=inv,
                            hh=float(res[:, 0] @ res[:, 0]), hi=float(res[:, 0] @ res[:, 1]),
                            ii=float(res[:, 1] @ res[:, 1]), ee=float(res[:, 2] @ res[:, 2]), dof=len(t) - 2)
        stats['mri'] = rec
        cn = self.vis.groupby('RID').dx.agg(lambda s: (s == 1).all())
        c = self.scans[self.scans.RID.isin(cn[cn].index) & self.scans.icv.notna()]
        stats['cn'] = c.sort_values('day').drop_duplicates('RID')[['RID', 'hippo', 'icv']].set_index('RID')
        return stats

    def fit_params(self, rids):
        """Hiệu chỉnh ICV + tham số mô hình hỗn hợp, chỉ từ các RID cho trước."""
        rids = set(rids)
        st = self._stats
        cn = st['cn'][st['cn'].index.isin(rids)]
        b = float(np.polyfit(cn.icv, cn.hippo, 1)[0])
        m = float(cn.icv.mean())
        out = dict(icv=dict(b=b, icv_mean=m, n=int(len(cn))))

        def agg(coefs, invs, rss, dof):
            coefs = np.array(coefs)
            s2 = rss / dof
            D = np.cov(coefs.T) - s2 * np.mean(invs, axis=0)
            w, V = np.linalg.eigh(D)
            D = V @ np.diag(np.maximum(w, 1e-6 * max(w.max(), 1e-12))) @ V.T
            return dict(gamma=coefs.mean(0).tolist(), D=D.tolist(), sigma2=float(s2), n=int(len(coefs)))

        for name in ['adas13', 'faq']:
            r = [v for k, v in st[name].items() if k in rids]
            out[name] = agg([v['coef'] for v in r], [v['inv'] for v in r], sum(v['rss'] for v in r), sum(v['dof'] for v in r))
        r = [v for k, v in st['mri'].items() if k in rids]
        ch = [v['coef_h'] - b * v['coef_i'] + np.array([b * m, 0.0]) for v in r]
        rss_h = sum(v['hh'] - 2 * b * v['hi'] + b * b * v['ii'] for v in r)
        dof = sum(v['dof'] for v in r)
        out['hippo_adj'] = agg(ch, [v['inv'] for v in r], rss_h, dof)
        out['entor'] = agg([v['coef_e'] for v in r], [v['inv'] for v in r], sum(v['ee'] for v in r), dof)
        return out


# =========================================================================== đặc trưng
def blup_slope(t, y, p):
    X = np.c_[np.ones_like(t), t]
    g, D = np.array(p['gamma']), np.array(p['D'])
    V = X @ D @ X.T + p['sigma2'] * np.eye(len(y))
    u = D @ X.T @ np.linalg.solve(V, y - X @ g)
    return float(g[1] + u[1])


def raw_points(src, points, lag=0, mri_window=MRI_MAX_AGE, series_key='version'):
    """Đầu vào thô tại mỗi mốc. lag: MRI phải có trước mốc >= lag ngày (S8)."""
    rows = []
    for r in points.itertuples():
        s = r.t0
        f = dict(RID=r.RID, t0=s)
        d = src._demo_by.get(r.RID)
        if d is not None:
            dd = d[d.day.notna() & (d.day <= s)].sort_values('day')
            for c in ['sex', 'education']:
                v = dd[c].dropna()
                f[c] = float(v.iloc[-1]) if len(v) else np.nan
            by = d.birth_year.dropna().unique()
            f['age'] = s.year - by[0] if len(by) == 1 and 18 <= s.year - by[0] <= 115 else np.nan
        for name, groups in src._by.items():
            g = groups.get(r.RID)
            if g is None:
                continue
            g = g[g.day <= s]
            if len(g) and (s - g.day.iloc[-1]).days <= COG_MAX_AGE:
                f[name] = g.value.iloc[-1]
                f[name + '_date'] = g.day.iloc[-1]
                if name in ('adas13', 'faq') and len(g) >= 2 and (g.day.iloc[-1] - g.day.iloc[0]).days >= MIN_SPAN:
                    f['ser_' + name] = (years(g.day.to_numpy(), g.day.to_numpy()[0]), g.value.to_numpy(float))
                    f[name + '_from'] = g.day.iloc[0]
        g = src._scans_by.get(r.RID)
        if g is not None:
            g = g[g.day <= s - pd.Timedelta(days=lag)] if lag else g[g.day <= s]
            if len(g) and (s - g.day.iloc[-1]).days <= mri_window:
                cur = g.iloc[-1]
                f.update(hippo=cur.hippo, icv=cur.icv, entor=cur.entor, field3t=cur.field3t,
                         mri_age_days=(s - cur.day).days, mri_date=cur.day, qc_done=cur.qc_done, scanner=cur.model,
                         mri_uid=cur.IMAGEUID, mri_run=cur.run)
                same = g[g[series_key] == cur[series_key]]
                span = (same.day.iloc[-1] - same.day.iloc[0]).days
                f['mri_span_days'] = span
                if span >= MIN_SPAN and same.icv.notna().all():
                    f['ser_mri'] = (years(same.day.to_numpy(), same.day.to_numpy()[0]), same.hippo.to_numpy(float),
                                    same.icv.to_numpy(float), same.entor.to_numpy(float))
                    f['mri_from'] = same.day.iloc[0]
                old = same[same.day <= cur.day - pd.Timedelta(days=TWO_POINT_MIN)]
                if len(old):
                    o = old.iloc[-1]
                    dd_ = (cur.day - o.day).days
                    f['hippo_rate2'] = 365.25 * (cur.hippo - o.hippo) / dd_
                    f['entor_rate2'] = 365.25 * (cur.entor - o.entor) / dd_
                    f['pair_days'] = dd_
        b = src._bsi_by.get(r.RID)
        if b is not None:
            b = b[b.day <= s]
            if len(b) and (s - b.day.iloc[-1]).days <= mri_window:
                f['hbsi_rate'] = b.hbsi_rate.iloc[-1]
        rows.append(f)
    raw = pd.DataFrame(rows)
    keep = [c for c in points.columns if c not in raw.columns]
    out = pd.concat([points[keep].reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    for c in ['sex', 'education', 'age', 'mmse', 'faq', 'adas13', 'ravlt_delay', 'hippo', 'icv', 'entor', 'field3t',
              'mri_age_days', 'mri_span_days', 'hippo_rate2', 'entor_rate2', 'hbsi_rate', 'ser_adas13', 'ser_faq', 'ser_mri']:
        if c not in out:
            out[c] = np.nan if not c.startswith('ser_') else None
    return out


def apply_params(raw, params):
    """Thêm hippo_adj và các slope BLUP theo tham số đã ước lượng (trong fold hoặc toàn train)."""
    out = raw.copy()
    b, m = params['icv']['b'], params['icv']['icv_mean']
    out['hippo_adj'] = out.hippo - b * (out.icv - m)
    hs, es, a_s, fs = [], [], [], []
    for sm, sa, sf in zip(out.ser_mri, out.ser_adas13, out.ser_faq):
        if isinstance(sm, tuple):
            t, h, i, e = sm
            hs.append(blup_slope(t, h - b * (i - m), params['hippo_adj']))
            es.append(blup_slope(t, e, params['entor']))
        else:
            hs.append(np.nan); es.append(np.nan)
        a_s.append(blup_slope(*sa, params['adas13']) if isinstance(sa, tuple) else np.nan)
        fs.append(blup_slope(*sf, params['faq']) if isinstance(sf, tuple) else np.nan)
    out['hippo_slope'], out['entor_slope'], out['adas13_slope'], out['faq_slope'] = hs, es, a_s, fs
    return out


def labelled_raw(src, confirmed=False, **kw):
    pts = attach_labels(eligible_points(src.vis), src.vis, confirmed=confirmed)
    return raw_points(src, pts, **kw)


def public(frame):
    """Bỏ các cột chuỗi đo nội bộ trước khi ghi CSV."""
    return frame[[c for c in frame.columns if not c.startswith('ser_')]]


def save_json(obj, path):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, allow_nan=False, ensure_ascii=False, default=str), encoding='utf8')
    tmp.replace(path)
