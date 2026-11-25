"""Kiểm thử hồi quy cho các lỗi đã sửa ở ver5. Chạy: python tests/run_tests.py"""
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
    print(('ĐẠT  ' if cond else 'LỖI  ') + name + (f' — {detail}' if detail else ''))


# 1. Lượt MCI mới nhất được dùng làm đầu vào dự báo (lỗi 3)
pts = eligible_points(src.vis)
last = src.vis.groupby('RID').tail(1)
last_mci = last[(last.dx == 2) & ~last.screen]
rid = int(last_mci.RID.iloc[0])
mine = pts[pts.RID == rid]
check('Lượt MCI cuối có trong đầu vào dự báo', (mine.t0.max() == last_mci.day.iloc[0]) and mine.has_followup.iloc[-1] == 0,
      f'RID {rid}, ngày {last_mci.day.iloc[0].date()}')
lab = attach_labels(pts, src.vis)
check('Lượt cuối không có nhãn (không dùng để train/test)', not ((lab.RID == rid) & (lab.t0 == last_mci.day.iloc[0])).any())

# 2. Tham số tiền xử lý chỉ dùng RID được phép (lỗi 1)
all_rids = set(src.vis.RID)
held = set(list(src._stats['adas13'])[:100])
p_all, p_sub = src.fit_params(all_rids), src.fit_params(all_rids - held)
check('Bỏ RID đánh giá làm giảm đúng số người ước lượng BLUP',
      p_all['adas13']['n'] - p_sub['adas13']['n'] == len(held), f"{p_all['adas13']['n']} → {p_sub['adas13']['n']}")

# 3. Giữ cận phải gốc khi khoảng vắt qua 1.095 ngày (lỗi 5)
check('Có khoảng L < 1095 < R hữu hạn', ((lab.L < 1095) & (lab.R > 1095) & np.isfinite(lab.R)).any())

# 4. Nhân khẩu lấy theo ngày mốc; xung đột được ghi lại (lỗi 2, 6)
check('Xung đột được ghi vào conflicts', len(src.conflicts) > 0, f'{len(src.conflicts)} bản ghi')
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
    check('Học vấn lấy theo bản ghi <= mốc', r.education.iloc[0] == found[2],
          f'RID {found[0]}: tại mốc {found[1].date()} dùng {found[2]}, bản ghi sau mốc là {found[3]}')
else:
    check('Học vấn lấy theo bản ghi <= mốc (không có ca đổi giá trị sau mốc để thử)', True)

# 5. MRI trùng ngày: mỗi RID-ngày còn đúng một dòng
check('UCSFFSX7: một dòng mỗi RID-ngày', not src.scans.duplicated(['RID', 'day']).any())

# 6. Upsert theo (RID, ngày mốc)
db = sqlite3.connect(':memory:')
db.execute('CREATE TABLE profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0)')
frame = pd.DataFrame([dict(RID=1, t0=pd.Timestamp('2020-01-01')), dict(RID=1, t0=pd.Timestamp('2021-01-01'))])
a1 = registry.upsert(db, registry.records(frame.iloc[:1], 'ZIP'))
a2 = registry.upsert(db, registry.records(frame, 'ZIP'))
a3 = registry.upsert(db, registry.records(frame, 'ZIP'))
check('Nhập lại thêm mốc mới, cập nhật mốc cũ', a1 == (1, 0) and a2 == (1, 0) and a3 == (0, 1), f'{a1} {a2} {a3}')

# 7. Gradient giải tích của mô hình khoảng
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
check('Gradient khớp sai phân hữu hạn', np.abs(g - num).max() / np.abs(num).max() < 1e-5)

print(f'\n{sum(ok)}/{len(ok)} kiểm thử đạt')
sys.exit(0 if all(ok) else 1)
