"""Chấm tập test 20% bằng các mô hình đã khóa từ train. Không học lại trên test.

Cách dùng:  python evaluate.py <thư_mục_test_đã_giải_nén>
Chỉ lưu kết quả tổng hợp (không lưu dữ liệu cá nhân của test) vào runs/.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import ROOT, MODELS, HORIZONS, build, read, has_table
from model import predict
from metrics import status, summarize, km_calibration
from validation import paired_deltas, PAIRS

MIN_SUBGROUP = 60
TABLES = ['DXSUM', 'MMSE', 'FAQ', 'PTDEMOG', 'UCSFFSX7', 'ADAS', 'NEUROBAT']


def roster(root, tables=TABLES):
    ids = set()
    for t in tables:
        if has_table(root, t):
            ids.update(read(root, t).RID.dropna().astype(int))
    return ids


def evaluate(testdir, replicates=20, seed=2026, save=True):
    testdir = Path(testdir)
    overlap = roster(ROOT / 'data' / 'train') & roster(testdir)
    if overlap:
        raise ValueError(f'{len(overlap)} RID trùng giữa train và test — dừng đánh giá')
    prep = json.loads((ROOT / 'runs' / 'preprocessing.json').read_text())
    models = json.loads((ROOT / 'runs' / 'models.json').read_text())
    secondary = json.loads((ROOT / 'runs' / 'secondary_models.json').read_text())
    df, _ = build(testdir, prep)
    shared = df.dropna(subset=MODELS['M3']).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(replicates):
        scopes = {'random': shared.sample(frac=1, random_state=int(rng.integers(1e9))).drop_duplicates('RID')}
        if (shared.second_mci == 1).sum() >= MIN_SUBGROUP:     # phân nhóm quá nhỏ thì không báo cáo
            scopes['second_mci'] = shared[shared.second_mci == 1]
        for scope, ev in scopes.items():
            for h in HORIZONS:
                y, w = status(ev, h, models['M0'], np.random.default_rng(rng.integers(1e9)), models['censoring'])
                pn = predict(models['null'], ev, h)
                for name in MODELS:
                    rows.append(pd.DataFrame(dict(repeat=r, scope=scope, RID=ev.RID.to_numpy(), h=h, model=name,
                                                  y=y, w=w, p=predict(models[name], ev, h), p_null=pn,
                                                  L=ev.L.to_numpy(), R=ev.R.to_numpy())))
    long = pd.concat(rows, ignore_index=True)
    per = []
    for (scope, rep, h, name), g in long.groupby(['scope', 'repeat', 'h', 'model']):
        per.append(dict(scope=scope, repeat=rep, h=h, model=name,
                        **summarize(g.y.to_numpy(), g.w.to_numpy(), g.p.to_numpy(), g.p_null.to_numpy(),
                                    g.L.to_numpy(), g.R.to_numpy(), h)))
    per = pd.DataFrame(per)
    num = per.select_dtypes('number').columns.drop(['repeat', 'h'])
    metrics = per.groupby(['scope', 'h', 'model'])[list(num)].mean().reset_index()
    deltas = paired_deltas(long, PAIRS)
    if (long.scope == 'second_mci').any():
        second = paired_deltas(long, [('M3', 'M2')], scope='second_mci', B=1000)
        second['comparison'] = second.comparison + ' (mốc MCI lần 2)'
        deltas = pd.concat([deltas, second], ignore_index=True)
    # bảng hiệu chuẩn theo nhóm (một mốc/người, lần lặp 0)
    cal = []
    ev0 = long[(long.scope == 'random') & (long.repeat == 0)]
    for (h, name), g in ev0.groupby(['h', 'model']):
        oe, slope, pred, obs = km_calibration(g.L.to_numpy(), g.R.to_numpy(), g.p.to_numpy(), h)
        for q, (a, b) in enumerate(zip(pred, obs)):
            cal.append(dict(h=h, model=name, group=q + 1, predicted=a, observed_km=b))
    # mô hình phụ: mỗi mô hình trên tập đủ biến riêng (độ bao phủ)
    cover = []
    for name, cols in MODELS.items():
        own = df.dropna(subset=cols)
        ev = own.sample(frac=1, random_state=seed).drop_duplicates('RID')
        for h in HORIZONS:
            y, w = status(ev, h, models['M0'], np.random.default_rng(seed), models['censoring'])
            s = summarize(y, w, predict(secondary[name], ev, h), predict(models['null'], ev, h),
                          ev.L.to_numpy(), ev.R.to_numpy(), h)
            cover.append(dict(model=name, h=h, persons=ev.RID.nunique(), **s))
    info = dict(test_landmarks=len(df), test_persons=int(df.RID.nunique()),
                shared_landmarks=len(shared), shared_persons=int(shared.RID.nunique()),
                shared_events_in_window=int(shared.drop_duplicates('RID').event.sum()),
                second_mci_persons=int((shared.second_mci == 1).sum()),
                note_second_mci='Không báo cáo phân nhóm mốc MCI lần 2 vì dưới %d người' % MIN_SUBGROUP
                if (shared.second_mci == 1).sum() < MIN_SUBGROUP else '')
    if save:
        out = ROOT / 'runs'
        metrics.to_csv(out / 'test_metrics.csv', index=False)
        deltas.to_csv(out / 'test_deltas.csv', index=False)
        pd.DataFrame(cal).to_csv(out / 'test_calibration.csv', index=False)
        pd.DataFrame(cover).to_csv(out / 'test_coverage.csv', index=False)
        (out / 'test_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf8')
    return metrics, deltas, info


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('test_directory', type=Path)
    a = p.parse_args()
    m, d, info = evaluate(a.test_directory)
    print(info)
    print(m[m.scope == 'random'][['h', 'model', 'n', 'cases', 'auc', 'ipa', 'oe_km', 'cal_slope_km', 'nb_core']]
          .round(4).to_string(index=False))
    print(d.round(4).to_string(index=False))
