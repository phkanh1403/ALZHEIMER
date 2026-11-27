"""ADNI data pipeline: cleaned sources, visit landmarks, interval-censored labels and as-of-visit features.

Three separate steps:
  1. Sources          read the tables of one data folder, resolve duplicates and conflicts (each one is logged),
                      and precompute per-participant statistics for the preprocessing parameters.
  2. raw_points()     PREDICTION inputs at every MCI visit, including the latest one (no later visit needed).
     attach_labels()  outcome intervals (L, R] only for visits with a later diagnosis (training and testing).
  3. Sources.fit_params() / apply_params()
                      head-size adjustment and BLUP trend parameters, estimated on a given set of participants
                      (within each cross-validation fold) and applied to every visit.

Every feature at visit s uses only data dated on or before s.
Interval notation: L = last day known dementia-free, R = first day dementia was recorded (inf if never), in days
after the visit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COG_MAX_AGE,
    DAYS_PER_YEAR,
    LME_MIN_POINTS,
    LME_MIN_SPAN,
    MIN_SPAN,
    MRI_MAX_AGE,
    TWO_POINT_MIN,
    WINDOW,
)

DX_NORMAL, DX_MCI, DX_DEMENTIA = 1, 2, 3
TREND_SERIES = ('adas13', 'faq')         # cognitive series with a BLUP trend
_CLINICAL = [  # (table, column, feature, valid range)
    ('MMSE', 'MMSCORE', 'mmse', 0, 30), ('FAQ', 'FAQTOTAL', 'faq', 0, 30),
    ('ADAS', 'TOTAL13', 'adas13', 0, 85), ('NEUROBAT', 'AVDEL30MIN', 'ravlt_delay', 0, 15),
]
_FS_COLUMNS = ['ST29SV', 'ST88SV', 'ST24TA', 'ST83TA', 'ST10CV']   # FreeSurfer: L/R hippocampus, L/R entorhinal, ICV
_MAD_TO_SD = 0.6745                      # median absolute deviation of a normal variable, in SDs
_RAW_COLUMNS = ['sex', 'education', 'age', 'mmse', 'faq', 'adas13', 'ravlt_delay', 'hippo', 'icv', 'entor',
                'field3t', 'mri_age_days', 'mri_span_days', 'hippo_rate2', 'entor_rate2', 'hbsi_rate',
                'ser_adas13', 'ser_faq', 'ser_mri']


# =========================================================================== table helpers
def _table_files(root, name):
    return list(Path(root).rglob(name + '.csv'))


def read_table(root, name):
    """Read the single <name>.csv under root (ADNI exports are Latin-1)."""
    files = _table_files(root, name)
    if len(files) != 1:
        raise ValueError(f'Expected exactly one {name}.csv, found {len(files)}')
    # copy() consolidates the column blocks of very wide tables (avoids pandas fragmentation warnings)
    return pd.read_csv(files[0], encoding='latin1', low_memory=False).copy()


def has_table(root, name):
    return len(_table_files(root, name)) == 1


def dated(df, column):
    """Add a parsed `day` column and drop rows without a valid date."""
    return df.assign(day=pd.to_datetime(df[column], errors='coerce')).dropna(subset=['day'])


def years(a, b):
    return (a - b) / np.timedelta64(1, 'D') / DAYS_PER_YEAR


def _days(a, b):
    return (a - b) / np.timedelta64(1, 'D')


def _by_rid(frame):
    return dict(tuple(frame.groupby('RID')))


# =========================================================================== diagnoses and landmarks
def visits(dx):
    """One row per participant and diagnosis day; days with conflicting diagnoses are removed and returned."""
    dx = dated(dx, 'EXAMDATE')
    dx['screen'] = (dx.VISCODE.fillna('').astype(str).str.lower().str.startswith('sc') |
                    dx.VISCODE2.fillna('').astype(str).str.lower().str.startswith('sc'))
    v = dx[dx.DIAGNOSIS.isin([DX_NORMAL, DX_MCI, DX_DEMENTIA])]
    n = v.groupby(['RID', 'day']).DIAGNOSIS.nunique()
    bad = n[n > 1].reset_index()[['RID', 'day']]
    v = v[~v.set_index(['RID', 'day']).index.isin(pd.MultiIndex.from_frame(bad))]
    out = v.groupby(['RID', 'day']).agg(dx=('DIAGNOSIS', 'first'), screen=('screen', 'all')).reset_index()
    return out.sort_values(['RID', 'day']).reset_index(drop=True), bad


def eligible_points(vis):
    """Every non-screening MCI visit with no prior dementia, INCLUDING the latest one (prediction input)."""
    rows = []
    for rid, h in vis.groupby('RID', sort=False):
        days, dxs, screen = h.day.to_numpy(), h.dx.to_numpy(), h.screen.to_numpy()
        mci_idx = np.where((dxs == DX_MCI) & ~screen)[0]
        if not len(mci_idx):
            continue
        dementia_so_far = np.cumsum(dxs == DX_DEMENTIA)
        for i in mci_idx:
            if dementia_so_far[i] > 0:
                continue
            is_second_mci = len(mci_idx) > 1 and i == mci_idx[1]
            rows.append((int(rid), pd.Timestamp(days[i]), years(days[i], days[mci_idx[0]]),
                         int(is_second_mci), int(i < len(dxs) - 1)))
    p = pd.DataFrame(rows, columns=['RID', 't0', 'tau', 'second_mci', 'has_followup'])
    p['tau2'] = p.tau ** 2
    return p


def _interval(days, dxs, visit, confirmed):
    """(L, R] in days after `visit`, or None when there is no later diagnosis."""
    later = np.where(days > visit)[0]
    if not len(later):
        return None
    dementia = [j for j in later if dxs[j] == DX_DEMENTIA]
    if confirmed:
        dementia = [j for j in dementia if (dxs[j:] == DX_DEMENTIA).all()]
    if dementia:
        upper = _days(days[dementia[0]], visit)
        before = [j for j in later if j < dementia[0]]
        lower = _days(days[before[-1]], visit) if before else 0.0
    else:
        upper, lower = np.inf, _days(days[later[-1]], visit)
    if lower >= WINDOW:
        lower, upper = float(WINDOW), np.inf
    return float(lower), float(upper)


def attach_labels(points, vis, confirmed=False):
    """Outcome interval (L, R] for visits with a later visit.

    Original bounds are kept when L < WINDOW; when L >= WINDOW the visit is censored at WINDOW.
    confirmed=True counts dementia only if every later diagnosis is also dementia (sensitivity analysis S4).
    """
    history = _by_rid(vis)
    lower, upper = [], []
    for r in points.itertuples():
        h = history[r.RID]
        bounds = _interval(h.day.to_numpy(), h.dx.to_numpy(), np.datetime64(r.t0), confirmed)
        lo, hi = bounds if bounds else (np.nan, np.nan)
        lower.append(lo)
        upper.append(hi)
    out = points.copy()
    out['L'], out['R'] = lower, upper
    out = out[out.L.notna() & ~((out.L <= 0) & ~np.isfinite(out.R))].copy()
    out['event'] = np.isfinite(out.R).astype(int)
    return out.reset_index(drop=True)


# =========================================================================== statistics helpers
def _ols(t, ys):
    """Least squares of several series on (1, t): coefficients (2 x k), (X'X)^-1 and residuals."""
    X = np.c_[np.ones_like(t), t]
    xtx_inv = np.linalg.inv(X.T @ X)
    B = xtx_inv @ X.T @ np.array(ys).T
    res = np.array(ys).T - X @ B
    return B, xtx_inv, res


def _pool_mixed_model(coefs, invs, rss, dof):
    """Random intercept-and-slope parameters by method of moments from per-participant OLS fits.

    D = Cov(beta_i) - sigma2 * mean((X_i'X_i)^-1), projected to the positive-definite cone.
    """
    coefs = np.array(coefs)
    s2 = rss / dof
    D = np.cov(coefs.T) - s2 * np.mean(invs, axis=0)
    w, V = np.linalg.eigh(D)
    D = V @ np.diag(np.maximum(w, 1e-6 * max(w.max(), 1e-12))) @ V.T
    return dict(gamma=coefs.mean(0).tolist(), D=D.tolist(), sigma2=float(s2), n=len(coefs))


def blup_slope(t, y, p):
    """Best linear unbiased predictor of one participant's slope, given mixed-model parameters p."""
    X = np.c_[np.ones_like(t), t]
    g, D = np.array(p['gamma']), np.array(p['D'])
    V = X @ D @ X.T + p['sigma2'] * np.eye(len(y))
    u = D @ X.T @ np.linalg.solve(V, y - X @ g)
    return float(g[1] + u[1])


# =========================================================================== data sources
class Sources:
    """Read and clean one data folder; every conflict is logged in self.conflicts.

    The *_at() methods return what was known about a participant on a given day.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.conflicts = []
        self.vis, bad = visits(read_table(root, 'DXSUM'))
        for r in bad.itertuples():
            self._log('DXSUM', r.RID, r.day.date(), 'conflicting diagnosis labels on the same day', 'visit removed')
        self.clin = self._clinical()
        self.scans, self.meas_error = self._mri()
        self.bsi = self._bsi()
        self.demo, self.demo_conflict = self._demographics()
        self._clin_by = {name: _by_rid(series) for name, series in self.clin.items()}
        self._scans_by = _by_rid(self.scans)
        self._bsi_by = _by_rid(self.bsi)
        self._demo_by = _by_rid(self.demo)
        self.series_stats = self._series_stats()

    def _log(self, table, rid, day, detail, action):
        self.conflicts.append(dict(table=table, RID=rid, day=day, detail=detail, action=action))

    # ---------------------------------------------------------------- clinical
    def _clinical(self):
        out = {}
        for table, col, name, lo, hi in _CLINICAL:
            if not has_table(self.root, table):
                continue
            d = read_table(self.root, table)
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
                self._log(table, rid, day.date(), f'{col}={vals}', 'measurement removed')
            s = s[~s.set_index(['RID', 'day']).index.isin(bad)].drop_duplicates(['RID', 'day'])
            out[name] = s.sort_values(['RID', 'day']).reset_index(drop=True)
        return out

    # ---------------------------------------------------------------- MRI
    def _read_mri(self):
        m = dated(read_table(self.root, 'UCSFFSX7'), 'EXAMDATE')
        m[_FS_COLUMNS] = m[_FS_COLUMNS].apply(pd.to_numeric, errors='coerce')
        keep = (~m.OVERALLQC.isin(['Fail', 'Hippocampus Only']) & ~m.HIPPOQC.eq('Fail') &
                ~m.TEMPQC.eq('Fail') & m[_FS_COLUMNS[:4]].gt(0).all(axis=1))
        m = m[keep].copy()
        m['hippo'] = m.ST29SV + m.ST88SV
        m['entor'] = (m.ST24TA + m.ST83TA) / 2
        m['icv'] = m.ST10CV
        m['field'] = m.FIELD_STRENGTH.astype(str)
        m['field3t'] = m.field.str.startswith('3').astype(int)
        m['model'] = 'NA'
        if has_table(self.root, 'All_Subjects_Structural_MRI_Images'):
            a = read_table(self.root, 'All_Subjects_Structural_MRI_Images')[['image_id', 'mri_mfr', 'mri_mfr_model']]
            a = a.drop_duplicates('image_id').set_index('image_id')
            scanner = (a.mri_mfr.astype(str) + ' ' + a.mri_mfr_model.astype(str)).to_dict()
            m['model'] = m.IMAGEUID.map(scanner).fillna('NA')
        m['qc_done'] = m.OVERALLQC.isin(['Pass', 'Partial']).astype(int)
        m['run'] = pd.to_datetime(m.RUNDATE, errors='coerce')
        return m

    def _merge_same_day(self, rid, day, g, modal_3t):
        """One scan per participant-day. Returns the kept row and, for same-protocol repeats, the HV difference."""
        if g.field3t.nunique() > 1:
            self._log('UCSFFSX7', rid, day.date(),
                      f'same-day scans at different field strengths {sorted(g.field.unique())}, '
                      f'IMAGEUID={g.IMAGEUID.tolist()}',
                      f'kept {"3T" if modal_3t else "1.5T"} (most common for this participant)')
            g = g[g.field3t == modal_3t]
        if len(g) == 1:
            return g.iloc[0], None
        diff = None
        if g.hippo.nunique() > 1 and g.FSVER.nunique() == 1:
            diff = float(g.hippo.iloc[0] - g.hippo.iloc[1])
            self._log('UCSFFSX7', rid, day.date(),
                      f'{len(g)} back-to-back scans, same protocol, HV={g.hippo.round(0).tolist()}, '
                      f'IMAGEUID={g.IMAGEUID.tolist()}', 'averaged')
        row = g.iloc[0].copy()
        for c in ['hippo', 'entor', 'icv']:
            row[c] = g[c].mean()
        row['qc_done'] = int(g.qc_done.max())
        row['model'] = g.model.mode().iloc[0]
        return row, diff

    def _mri(self):
        m = self._read_mri()
        modal_3t = m.groupby('RID').field3t.agg(lambda s: int(s.mean() >= 0.5))   # ties -> 3T
        rows, diffs = [], []
        for (rid, day), g in m.groupby(['RID', 'day']):
            if len(g) == 1:
                rows.append(g.iloc[0])
                continue
            row, diff = self._merge_same_day(rid, day, g, modal_3t[rid])
            rows.append(row)
            if diff is not None:
                diffs.append(diff)
        s = pd.DataFrame(rows)[['RID', 'day', 'hippo', 'entor', 'icv', 'field', 'field3t', 'FSVER', 'model',
                                'qc_done', 'IMAGEUID', 'run']]
        s['version'] = s.FSVER.astype(str) + '|' + s.field
        s['version_model'] = s.version + '|' + s.model
        return s.sort_values(['RID', 'day']).reset_index(drop=True), _measurement_error(np.array(diffs))

    def _bsi(self):
        if not has_table(self.root, 'FOXLABBSI'):
            return pd.DataFrame(columns=['RID', 'day', 'hbsi_rate'])
        f = dated(read_table(self.root, 'FOXLABBSI'), 'EXAMDATE')
        base = (f[['LONIUID', 'day']].drop_duplicates('LONIUID')
                .rename(columns={'LONIUID': 'LONIUID_BASE', 'day': 'base_day'}))
        accepted = f.HBSI_L.notna() & f.HBSI_R.notna() & f.HPACCEPT_L.eq(1) & f.HPACCEPT_R.eq(1)
        h = f[accepted].merge(base, on='LONIUID_BASE')
        dt = (h.day - h.base_day).dt.days
        h = h[dt >= TWO_POINT_MIN].assign(dt=dt[dt >= TWO_POINT_MIN])
        h['hbsi_rate'] = -(h.HBSI_L + h.HBSI_R) * 1000 * DAYS_PER_YEAR / h.dt   # mm³/year
        return h.groupby(['RID', 'day'], as_index=False).hbsi_rate.mean()

    def _demographics(self):
        d = read_table(self.root, 'PTDEMOG')
        d = d.assign(day=pd.to_datetime(d.VISDATE, errors='coerce'))
        d['sex'] = pd.to_numeric(d.PTGENDER, errors='coerce').where(lambda s: s.between(1, 2))
        d['education'] = pd.to_numeric(d.PTEDUCAT, errors='coerce').where(lambda s: s.between(0, 36))
        year = r'((?:18|19|20)\d{2})'
        y = pd.to_numeric(d.PTDOBYY.astype(str).str.extract(year)[0], errors='coerce')
        if 'PTDOB' in d:
            y = y.fillna(pd.to_numeric(d.PTDOB.astype(str).str.extract(year)[0], errors='coerce'))
        d['birth_year'] = y
        conflict = set()
        for col in ['sex', 'education']:
            n = d.dropna(subset=[col]).groupby('RID')[col].nunique()
            bad = set(n[n > 1].index.astype(int))
            conflict |= bad
            for rid in sorted(bad):
                vals = sorted(d.loc[d.RID == rid, col].dropna().unique().tolist())
                self._log('PTDEMOG', rid, '', f'{col} differs between records: {vals}',
                          'latest record on or before visit used; excluded in S7')
        return d[['RID', 'day', 'sex', 'education', 'birth_year']], conflict

    # ---------------------------------------------------------------- statistics for preprocessing parameters
    def _series_stats(self):
        """Per-participant OLS for each series; population parameters for any participant set are a pooling."""
        mci = set(self.vis.loc[self.vis.dx == DX_MCI, 'RID'])
        stats = {}

        def usable_time(g):
            if len(g) < LME_MIN_POINTS:
                return None
            t = years(g.day.to_numpy(), g.day.to_numpy()[0])
            return t if t[-1] * DAYS_PER_YEAR >= LME_MIN_SPAN else None

        for name in TREND_SERIES:
            rec = {}
            for rid, g in self.clin[name][self.clin[name].RID.isin(mci)].groupby('RID'):
                t = usable_time(g)
                if t is None:
                    continue
                B, inv, res = _ols(t, [g.value.to_numpy(float)])
                rec[rid] = dict(coef=B[:, 0], inv=inv, rss=float(res[:, 0] @ res[:, 0]), dof=len(t) - 2)
            stats[name] = rec
        rec = {}
        for rid, scans in self.scans[self.scans.RID.isin(mci)].groupby('RID'):
            g = scans[scans.version == scans.version.mode().iloc[0]]
            if g.icv.isna().any():
                continue
            t = usable_time(g)
            if t is None:
                continue
            B, inv, res = _ols(t, [g.hippo.to_numpy(float), g.icv.to_numpy(float), g.entor.to_numpy(float)])
            rec[rid] = dict(coef_h=B[:, 0], coef_i=B[:, 1], coef_e=B[:, 2], inv=inv,
                            hh=float(res[:, 0] @ res[:, 0]), hi=float(res[:, 0] @ res[:, 1]),
                            ii=float(res[:, 1] @ res[:, 1]), ee=float(res[:, 2] @ res[:, 2]), dof=len(t) - 2)
        stats['mri'] = rec
        always_normal = self.vis.groupby('RID').dx.agg(lambda s: (s == DX_NORMAL).all())
        c = self.scans[self.scans.RID.isin(always_normal[always_normal].index) & self.scans.icv.notna()]
        stats['cn'] = c.sort_values('day').drop_duplicates('RID')[['RID', 'hippo', 'icv']].set_index('RID')
        return stats

    def fit_params(self, rids):
        """Head-size adjustment and mixed-model parameters, from the given participants only."""
        rids = set(rids)
        st = self.series_stats
        cn = st['cn'][st['cn'].index.isin(rids)]
        b = float(np.polyfit(cn.icv, cn.hippo, 1)[0])
        m = float(cn.icv.mean())
        out = dict(icv=dict(b=b, icv_mean=m, n=len(cn)))
        for name in TREND_SERIES:
            r = [v for k, v in st[name].items() if k in rids]
            out[name] = _pool_mixed_model([v['coef'] for v in r], [v['inv'] for v in r],
                                          sum(v['rss'] for v in r), sum(v['dof'] for v in r))
        r = [v for k, v in st['mri'].items() if k in rids]
        # adjusted hippocampus h - b (icv - m): coefficients and residual sum of squares follow by linearity
        coef_adj = [v['coef_h'] - b * v['coef_i'] + np.array([b * m, 0.0]) for v in r]
        rss_adj = sum(v['hh'] - 2 * b * v['hi'] + b * b * v['ii'] for v in r)
        dof = sum(v['dof'] for v in r)
        out['hippo_adj'] = _pool_mixed_model(coef_adj, [v['inv'] for v in r], rss_adj, dof)
        out['entor'] = _pool_mixed_model([v['coef_e'] for v in r], [v['inv'] for v in r],
                                         sum(v['ee'] for v in r), dof)
        return out

    # ---------------------------------------------------------------- as-of-visit queries
    def demographics_at(self, rid, day):
        d = self._demo_by.get(rid)
        if d is None:
            return {}
        known = d[d.day.notna() & (d.day <= day)].sort_values('day')
        out = {}
        for c in ['sex', 'education']:
            v = known[c].dropna()
            out[c] = float(v.iloc[-1]) if len(v) else np.nan
        birth = d.birth_year.dropna().unique()
        out['age'] = day.year - birth[0] if len(birth) == 1 and 18 <= day.year - birth[0] <= 115 else np.nan
        return out

    def clinical_at(self, rid, day):
        out = {}
        for name, groups in self._clin_by.items():
            g = groups.get(rid)
            if g is None:
                continue
            g = g[g.day <= day]
            if not len(g) or (day - g.day.iloc[-1]).days > COG_MAX_AGE:
                continue
            out[name] = g.value.iloc[-1]
            out[name + '_date'] = g.day.iloc[-1]
            if name in TREND_SERIES and len(g) >= 2 and (g.day.iloc[-1] - g.day.iloc[0]).days >= MIN_SPAN:
                out['ser_' + name] = (years(g.day.to_numpy(), g.day.to_numpy()[0]), g.value.to_numpy(float))
                out[name + '_from'] = g.day.iloc[0]
        return out

    def mri_at(self, rid, day, lag=0, window=MRI_MAX_AGE, series_key='version'):
        """Latest scan at least `lag` days before the visit and at most `window` days old, plus its series."""
        g = self._scans_by.get(rid)
        if g is None:
            return {}
        g = g[g.day <= day - pd.Timedelta(days=lag)]
        if not len(g) or (day - g.day.iloc[-1]).days > window:
            return {}
        cur = g.iloc[-1]
        out = dict(hippo=cur.hippo, icv=cur.icv, entor=cur.entor, field3t=cur.field3t,
                   mri_age_days=(day - cur.day).days, mri_date=cur.day, qc_done=cur.qc_done, scanner=cur.model,
                   mri_uid=cur.IMAGEUID, mri_run=cur.run)
        same = g[g[series_key] == cur[series_key]]
        span = (same.day.iloc[-1] - same.day.iloc[0]).days
        out['mri_span_days'] = span
        if span >= MIN_SPAN and same.icv.notna().all():
            out['ser_mri'] = (years(same.day.to_numpy(), same.day.to_numpy()[0]), same.hippo.to_numpy(float),
                              same.icv.to_numpy(float), same.entor.to_numpy(float))
            out['mri_from'] = same.day.iloc[0]
        earlier = same[same.day <= cur.day - pd.Timedelta(days=TWO_POINT_MIN)]
        if len(earlier):
            prev = earlier.iloc[-1]
            gap = (cur.day - prev.day).days
            out['hippo_rate2'] = DAYS_PER_YEAR * (cur.hippo - prev.hippo) / gap
            out['entor_rate2'] = DAYS_PER_YEAR * (cur.entor - prev.entor) / gap
            out['pair_days'] = gap
        return out

    def bsi_at(self, rid, day, window=MRI_MAX_AGE):
        b = self._bsi_by.get(rid)
        if b is None:
            return {}
        b = b[b.day <= day]
        if len(b) and (day - b.day.iloc[-1]).days <= window:
            return {'hbsi_rate': b.hbsi_rate.iloc[-1]}
        return {}


def _measurement_error(diffs):
    """Robust SD of a single hippocampal measurement from same-day repeat pairs."""
    if not len(diffs):
        return dict(pairs=0, median_abs_diff=np.nan, sigma_robust=np.nan)
    mad = float(np.median(np.abs(diffs)))
    return dict(pairs=len(diffs), median_abs_diff=mad, sigma_robust=float(mad / _MAD_TO_SD / np.sqrt(2)))


# =========================================================================== features
def raw_points(src, points, lag=0, mri_window=MRI_MAX_AGE, series_key='version'):
    """Raw inputs at each visit. lag: the MRI must precede the visit by at least `lag` days (S8)."""
    rows = []
    for r in points.itertuples():
        f = dict(RID=r.RID, t0=r.t0)
        f.update(src.demographics_at(r.RID, r.t0))
        f.update(src.clinical_at(r.RID, r.t0))
        f.update(src.mri_at(r.RID, r.t0, lag=lag, window=mri_window, series_key=series_key))
        f.update(src.bsi_at(r.RID, r.t0, window=mri_window))
        rows.append(f)
    raw = pd.DataFrame(rows)
    keep = [c for c in points.columns if c not in raw.columns]
    out = pd.concat([points[keep].reset_index(drop=True), raw.reset_index(drop=True)], axis=1)
    for c in _RAW_COLUMNS:
        if c not in out:
            out[c] = None if c.startswith('ser_') else np.nan
    return out


def apply_params(raw, params):
    """Add hippo_adj and BLUP slopes using the given parameters (within-fold or full training data)."""
    out = raw.copy()
    b, m = params['icv']['b'], params['icv']['icv_mean']
    out['hippo_adj'] = out.hippo - b * (out.icv - m)
    hippo, entor, adas, faq = [], [], [], []
    for mri, adas_series, faq_series in zip(out.ser_mri, out.ser_adas13, out.ser_faq, strict=True):
        if isinstance(mri, tuple):
            t, h, icv, e = mri
            hippo.append(blup_slope(t, h - b * (icv - m), params['hippo_adj']))
            entor.append(blup_slope(t, e, params['entor']))
        else:
            hippo.append(np.nan)
            entor.append(np.nan)
        adas.append(blup_slope(*adas_series, params['adas13']) if isinstance(adas_series, tuple) else np.nan)
        faq.append(blup_slope(*faq_series, params['faq']) if isinstance(faq_series, tuple) else np.nan)
    out['hippo_slope'], out['entor_slope'], out['adas13_slope'], out['faq_slope'] = hippo, entor, adas, faq
    return out


def labelled_raw(src, confirmed=False, **kw):
    """Raw inputs at labelled visits (training and testing)."""
    points = attach_labels(eligible_points(src.vis), src.vis, confirmed=confirmed)
    return raw_points(src, points, **kw)


def public(frame):
    """Drop internal measurement-series columns before writing CSV."""
    return frame[[c for c in frame.columns if not c.startswith('ser_')]]
