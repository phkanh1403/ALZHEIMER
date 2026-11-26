"""Regression tests for data handling and app logic. Run: python tests/run_tests.py"""
import io
import json
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import ROOT, MODELS, Sources, eligible_points, attach_labels, raw_points, apply_params, read  # noqa: E402
from model import fit_interval  # noqa: E402
import registry  # noqa: E402

DATA = ROOT / 'data' / 'train'
src = Sources(DATA)
ok = []


def check(name, cond, detail=''):
    ok.append(bool(cond))
    print(('PASS  ' if cond else 'FAIL  ') + name + (f' — {detail}' if detail else ''))


# 1. The latest MCI visit is used as a prediction input
pts = eligible_points(src.vis)
last = src.vis.groupby('RID').tail(1)
last_mci = last[(last.dx == 2) & ~last.screen]
rid = int(last_mci.RID.iloc[0])
mine = pts[pts.RID == rid]
check('Most recent MCI visit is available for prediction', (mine.t0.max() == last_mci.day.iloc[0]) and mine.has_followup.iloc[-1] == 0,
      f'RID {rid}, visit {last_mci.day.iloc[0].date()}')
lab = attach_labels(pts, src.vis)
check('Most recent visit has no outcome label (not used for training/testing)', not ((lab.RID == rid) & (lab.t0 == last_mci.day.iloc[0])).any())

# 2. Preprocessing parameters use only permitted participants
all_rids = set(src.vis.RID)
held = set(list(src._stats['adas13'])[:100])
p_all, p_sub = src.fit_params(all_rids), src.fit_params(all_rids - held)
check('Held-out participants are excluded from preprocessing parameters',
      p_all['adas13']['n'] - p_sub['adas13']['n'] == len(held), f"{p_all['adas13']['n']} → {p_sub['adas13']['n']}")

# 3. Right bound kept for intervals crossing day 1,095
check('Intervals crossing day 1095 keep their right bound', ((lab.L < 1095) & (lab.R > 1095) & np.isfinite(lab.R)).any())

# 4. Demographics as of the visit; conflicts logged
check('Data conflicts are logged', len(src.conflicts) > 0, f'{len(src.conflicts)} records')
found = None
for rid_c in sorted(src.demo_conflict):
    d = src._demo_by[rid_c].dropna(subset=['education', 'day']).sort_values('day')
    p = pts[pts.RID == rid_c]
    for t in p.t0:
        before, after = d[d.day <= t], d[d.day > t]
        if len(before) and len(after) and after.education.iloc[-1] != before.education.iloc[-1]:
            found = (rid_c, t, before.education.iloc[-1], after.education.iloc[-1]); break
    if found:
        break
if found:
    r = raw_points(src, pts[(pts.RID == found[0]) & (pts.t0 == found[1])])
    check('Education uses the record on or before the visit', r.education.iloc[0] == found[2],
          f'RID {found[0]}: at {found[1].date()} uses {found[2]}; later record says {found[3]}')
else:
    check('Education uses the record on or before the visit (no test case)', True)

# 5. Same-day MRI: one row per participant-day
check('UCSFFSX7: one row per participant-day', not src.scans.duplicated(['RID', 'day']).any())

# 6. Upsert by (RID, visit date)
db = sqlite3.connect(':memory:')
db.execute('CREATE TABLE profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0)')
frame = pd.DataFrame([dict(RID=1, t0=pd.Timestamp('2020-01-01')), dict(RID=1, t0=pd.Timestamp('2021-01-01'))])
a1 = registry.upsert(db, registry.records(frame.iloc[:1], 'ZIP'))
a2 = registry.upsert(db, registry.records(frame, 'ZIP'))
a3 = registry.upsert(db, registry.records(frame, 'ZIP'))
check('Re-import adds new visits and updates existing ones', a1 == (1, 0) and a2 == (1, 0) and a3 == (0, 1), f'{a1} {a2} {a3}')

# 7. Analytic gradient of the interval model
import scipy.optimize as so  # noqa: E402
import model as M  # noqa: E402
df = pd.read_csv(ROOT / 'runs' / 'train_landmarks.csv').dropna(subset=MODELS['M3'])
cap, orig = {}, so.minimize
M.minimize = lambda f, x0, **k: (cap.setdefault('f', f), orig(f, x0, **k))[1]
m = M.fit_interval(df, MODELS['M3'])
M.minimize = orig
f = cap['f']
x = np.r_[m['log_hazard'], m['beta']] + 0.05
v, g = f(x)
num = np.array([(f(x + e)[0] - f(x - e)[0]) / 2e-6 for e in np.eye(len(x)) * 1e-6])
check('Analytic gradient matches finite differences', np.abs(g - num).max() / np.abs(num).max() < 1e-5)

# 8. Manually entered patient: trends from history tables
prep = json.loads((ROOT / 'runs' / 'preprocessing.json').read_text(encoding='utf-8'))
cog = pd.DataFrame({'Date': ['2024-01-10', '2025-01-12'], 'ADAS-Cog13': [14.0, 16.0], 'FAQ': [1.0, 2.0]})
mri = pd.DataFrame({'Date': ['2024-01-15'], 'Hippocampal volume (mm³)': [6600.0], 'Intracranial volume (mm³)': [1.5e6],
                    'Entorhinal thickness (mm)': [3.3]})
f = registry.manual_features(prep, '2026-01-20', 2, 74, 'Female', 16, 26, 4, 19, 3, '2026-01-10', 6400.0, 1.5e6, 3.1, '3T', cog, mri)
check('Hand-entered patient gets all M3 inputs', all(f.get(c) is not None for c in MODELS['M3']) and f['mri_span_days'] >= 300,
      f"hippocampal change {f['hippo_slope']:+.0f} mm³/year")
short = registry.manual_features(prep, '2026-01-20', 2, 74, 'Female', 16, 26, 4, 19, 3, '2026-01-10', 6400.0, 1.5e6, 3.1, '3T',
                                 None, mri.assign(Date=['2025-10-01']))
check('MRI history < 300 days gives no atrophy trend', 'hippo_slope' not in short and short['mri_span_days'] < 300)

print(f'\n{sum(ok)}/{len(ok)} tests passed')
sys.exit(0 if all(ok) else 1)
